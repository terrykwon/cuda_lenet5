#include "LeNet5_cuda.h"



/**
 * Wrapper to catch CUDA errors.
 * For debugging only.
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

LeNet5_cuda::LeNet5_cuda(int batch) : LeNet5(batch) {
  this->f_conv1_weight = new float[conv1_in_channel * conv1_out_channel *
                                  conv1_kernel_size * conv1_kernel_size];
  this->f_conv1_bias = new float[conv1_out_channel];
  this->f_conv2_weight = new float[conv2_in_channel * conv2_out_channel *
                                  conv2_kernel_size * conv2_kernel_size];
  this->f_conv2_bias = new float[conv2_out_channel];
  this->f_fc1_weight = new float[fc1_in_channel * fc1_out_channel];
  this->f_fc1_bias = new float[fc1_out_channel];
  this->f_fc2_weight = new float[fc2_in_channel * fc2_out_channel];
  this->f_fc2_bias = new float[fc2_out_channel];
  this->f_fc3_weight = new float[fc3_in_channel * fc3_out_channel];
  this->f_fc3_bias = new float[fc3_out_channel];

  // Activation
  this->f_output = new float[batch * output_size];
}


/**
 * Each thread writes a (K*K) partial column.
 * Writing by columns is good b/c coalesces over rows.
 *
 * output: (C * K * K) x (H * W)
 */
__global__ void im2col(float* X, float* X_unrolled, int IC, int H, int W, int K) {
  int H_OUT = H - (K - 1); // output dimensions
  int W_OUT = W - (K - 1);

  int b = blockIdx.x;
  int t = blockIdx.y * 1024 + threadIdx.x;
  int W_prime = H_OUT * W_OUT;
  int H_prime = IC * K * K;

  if (t < IC*W_prime) { // each thread will update one channel of one column
    int ic = t / W_prime; // channel = row block
    int s = t % W_prime; // column

    int h_out = s / W_OUT; // original output row
    int w_out = s % W_OUT; // original output column

    int w_unroll = h_out * W_OUT + w_out; // index of 
    int h_base = ic * K * K; // starting row of unrolled

    for (int p = 0; p < K; p++) {
      for (int q = 0; q < K; q++) {
        int h_unroll = h_base + p*K + q;

        int in_idx = b*(IC*H*W) + ic*(H*W) + (h_out+p)*(W) + (w_out+q);
        int out_idx = b*(H_prime*W_prime) + h_unroll*(W_prime) + w_unroll;

        // printf("b=%d t=%d ic=%d h_out=%d w_out=%d h_unroll=%d w_unroll=%d out_idx=%d\n", 
        //     b, t, ic, h_out, w_out, h_unroll, w_unroll, out_idx);

        X_unrolled[out_idx] = X[in_idx];
        // X_unrolled[out_idx] = 1.0f;
      }
    }
  }
}

/**
 * This can be called in prepare() since it is just reordering
 * (IC x OC x K x K) to (OC x IC x K x K).
 */
void reorder_filters(double* F, float* F_col, int IC, int OC, int K) {
  for (int ic = 0; ic < IC; ic++) {
    for (int oc = 0; oc < OC; oc++) {
      for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
          F_col[oc*(IC*K*K) + ic*(K*K) + i*(K) + j] = float(F[ic*(OC*K*K) + oc*(K*K) + i*(K) + j]);
        }
      }
    }
  }
}


void unrolled_conv(float* X, float* X_unrolled, float* Y, float* weight, float* bias, 
    const int B, const int H, const int W, const int IC, const int OC, int K) {

  int H_OUT = H - (K - 1); // output dimensions
  int W_OUT = W - (K - 1);

  int total_threads = IC * H_OUT * W_OUT;
  dim3 unrollGridDim(B, ceil(total_threads / 1024), 1);
  dim3 unrollBlockDim(1024, 1, 1);
  im2col<<<unrollGridDim, unrollBlockDim>>>(X, X_unrolled, IC, H, W, K);

  // dim3 matmulGridDim();
  // dim3 matmulBlockDim();
  // naive_matmul_1(f_conv1_weight, X_unrolled, Y);
}

void unrolled_conv_1(float* X, float* X_unrolled, float* Y, 
    const int B, const int H, const int W, const int IC, const int OC, int K) {

  int H_OUT = H - (K - 1); // output dimensions
  int W_OUT = W - (K - 1);

  int total_threads = IC * H_OUT * W_OUT;
  dim3 unrollGridDim(B, ceil(total_threads / 1024), 1);
  dim3 unrollBlockDim(1024, 1, 1);
  im2col<<<unrollGridDim, unrollBlockDim>>>(X, X_unrolled, IC, H, W, K);

  dim3 matmulGridDim();
  dim3 matmulBlockDim();
  // naive_matmul_1(f_conv1_weight, X_unrolled, Y);
}

void unrolled_conv_2(float* X, float* X_unrolled, float* Y, 
    const int B, const int H, const int W, const int IC, const int OC, int K) {

  int H_OUT = H - (K - 1); // output dimensions
  int W_OUT = W - (K - 1);

  int total_threads = IC * H_OUT * W_OUT;
  dim3 unrollGridDim(B, ceil(total_threads / 1024), 1);
  dim3 unrollBlockDim(1024, 1, 1);
  im2col<<<unrollGridDim, unrollBlockDim>>>(X, X_unrolled, IC, H, W, K);

  // naive_matmul(X_unrolled, Y);
}

__global__ void conv1(float* input, float* output,
                      int B, const int H, const int W, const int IC, int OC,
                      int K) {
  int H_OUT = H - (K - 1); // output dimensions
  int W_OUT = W - (K - 1);

  int b = blockIdx.x; // batch
  int oc = blockIdx.y; // output channel
  int w = threadIdx.x; // col
  int h = threadIdx.y; // row

  __shared__ float X_shared[3*32*32]; // static allocation
  // We can load the entire input into shared memory?
  for (int channel = 0; channel < 3; channel++) {
    X_shared[channel*(H*W) + h*(W) + w] = input[b*(IC*H*W) + channel*(H*W) + h*(W) + w];
  }
  __syncthreads();

  if (w < W_OUT && h < H_OUT) { // more threads than output 
    // Convolution
    int output_index =
        b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * W_OUT + w;
    // output[output_index] = d_conv1_bias[oc];
    float val = d_conv1_bias[oc];

    for (int ic = 0; ic < IC; ic++) { // input channels
      // int input_base = b * (IC * H * W) + ic * (H * W) + h * (W) + w;
      int kernel_base = oc * (IC * K * K) + ic * (K * K);

      for (int kh = 0; kh < K; kh++) { // kernel height
        for (int kw = 0; kw < K; kw++) { // kernel width
          // float val = input[input_base + kh * (W) + kw] *
          //               d_conv1_weight[kernel_base + kh * (K) + kw];
          val += X_shared[ic*(H*W) + h*(W) + w + kh*(W) + kw] *
                        d_conv1_weight[kernel_base + kh * (K) + kw];
        }
      }
    }
    output[output_index] = val;
  }
}


__global__ void conv2(float* input, float* output,
                      int B, const int H, const int W, const int IC, int OC,
                      int K) {
  int H_OUT = H - (K - 1); // output dimensions
  int W_OUT = W - (K - 1);

  int b = blockIdx.x; // batch
  int oc = blockIdx.y; // output channel
  int w = threadIdx.x; // col
  int h = threadIdx.y; // row

  __shared__ float X_shared[6*14*14]; // static allocation
  // We can load the entire input into shared memory?
  for (int channel = 0; channel < 6; channel++) {
    X_shared[channel*(H*W) + h*(W) + w] = input[b*(IC*H*W) + channel*(H*W) + h*(W) + w];
  }
  __syncthreads();

  if (w < W_OUT && h < H_OUT) { // more threads than output 
    // Convolution
    int output_index =
        b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * W_OUT + w;
    // output[output_index] = d_conv2_bias[oc];
    float val = d_conv2_bias[oc];
    for (int ic = 0; ic < IC; ic++) { // input channels
      // int input_base = b * (IC * H * W) + ic * (H * W) + h * (W) + w;
      int kernel_base = oc * (IC * K * K) + ic * (K * K);

      for (int kh = 0; kh < K; kh++) { // kernel height
        for (int kw = 0; kw < K; kw++) { // kernel width
          // float val = input[input_base + kh * (W) + kw] *
          //               d_conv1_weight[kernel_base + kh * (K) + kw];
          val += X_shared[ic*(H*W) + h*(W) + w + kh*(W) + kw] *
                        d_conv2_weight[kernel_base + kh * (K) + kw];
        }
      }
    }
    output[output_index] = val;
  }
}


/**
 * (batch_size, 3, 1) x (32, 32, 1)
 */
__global__ void normalize(int batch, int input_channel, int input_size, const uint8_t* const d_image, float* d_input) {
  // automatically placed in registers. Should these be in shared / constant?
  // probably not because they're just single variables
  const float max_int = 255.0f;   
  const float mean = 0.5f;
  const float var = 0.5f;

  const int batch_id = blockIdx.x;
  const int channel_id = blockIdx.y;
  const int col = threadIdx.x;
  const int row = threadIdx.y;

  float val;

  if (col < input_size && row < input_size) {
    // standard normalize, center at 0
    // one global memory read, one write
    val = d_image[batch_id*input_channel*input_size*input_size + channel_id*input_size*input_size + row*input_size + col];
    val = ((val/max_int) - mean) / var;
    d_input[batch_id*input_channel*input_size*input_size + channel_id*input_size*input_size + row*input_size + col] = val;
  }
}

/**
 * (batch_size, out_channels, 1) x (width, height, 1)
 */
__global__ void naive_conv(float* input, float* output, float* weight,
                      float* bias, int B, int H, int W, int IC, int OC,
                      int K) {
  int H_OUT = H - (K - 1); // output dimensions
  int W_OUT = W - (K - 1);

  int b = blockIdx.x; // batch
  int oc = blockIdx.y; // output channel
  int w = threadIdx.x; // col
  int h = threadIdx.y; // row

  // Convolution
  int output_index =
      b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * W_OUT + w;
  output[output_index] = bias[oc];
  for (int ic = 0; ic < IC; ic++) {
    int input_base = b * (IC * H * W) + ic * (H * W) + h * (W) + w;
    int kernel_base = oc * (IC * K * K) + ic * (K * K);
    for (int kh = 0; kh < K; kh++) {
      for (int kw = 0; kw < K; kw++) {
        float val = input[input_base + kh * (W) + kw] *
                      weight[kernel_base + kh * (K) + kw];
        output[output_index] += val;
      }
    }
  }
}

/**
 * (batch_size, in_channels, 1) x (width, height, 1)
 * Element-wise. 
 */
__global__ void naive_relu(float* feature_map, int channels, int width, int height) {
  int b = blockIdx.x; // batch
  int oc = blockIdx.y; // output channel
  int w = threadIdx.x; // col
  int h = threadIdx.y; // row

  int index = b * (channels * width * height) + oc * (width * height) + h * width + w;

  feature_map[index] = fmax(feature_map[index], 0.0f);
}

/**
 * (batch_size, in_channels, 1) x (width, height, 1)
 */
__global__ void naive_pool(float* input, float* output, int C, int H, int W) {
  int scale = 2;
  int H_OUT = H / scale;
  int W_OUT = W / scale;

  int b = blockIdx.x; // batch
  int c = blockIdx.y; // output channel
  int w = threadIdx.x; // col
  int h = threadIdx.y; // row

  int input_base = b * (C * H * W) + c * (H * W) + (h*2) * (W) + (w*2);
  int max_sh = 0;
  int max_sw = 0;
  float max_val = 0.0f; // since after relu

  // Find maximum
  for (int sh = 0; sh < scale; sh++) {
    for (int sw = 0; sw < scale; sw++) {
      float val = input[input_base + sh * (W) + sw];
      if (val > max_val) {
        max_val = val;
      }
    }
  }

  // Set output with max value
  int output_index = b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) +
                      h * (W_OUT) + w;
  output[output_index] = max_val;
}

/**
 * (batch, 1, 1) x (output_nodes, 1, 1)
 */
__global__ void naive_fc(float* input, float* output, float* weight, float* bias,
                         int IC, int OC) {
  int b = blockIdx.x;
  int oc = threadIdx.x; 

  // output[b * OC + oc] = bias[oc];
  float val = bias[oc];
  for (int ic = 0; ic < IC; ic++)
    val += weight[oc * IC + ic] * input[b * IC + ic];

  output[b*OC + oc] = val;
}

/**
 * (batch, num_rowblocks, 1) x (output_nodes, 1, 1)
 */
__global__ void fc_rowblock(float* input, float* output, float* weight, float* bias,
                         int IC, int OC, int num_rowblocks) {
  int batch = blockIdx.x;
  int block = blockIdx.y;
  int t = threadIdx.x; 

  int block_length = OC / num_rowblocks; // assume divisible
  int row_start = block * IC * block_length;

  extern __shared__ float shmem[]; // stores one complete row block of W

  if (t < IC) {
    for (int r = 0; r < block_length; r++) {
      shmem[r*IC + t] = weight[row_start + r*IC + t];
    }
  }
  __syncthreads();

  if (t < block_length) { // calculate one output element of y
    float val = bias[block*block_length + t];

    #pragma unroll
    for (int ic = 0; ic < IC; ic++) {
      val += shmem[t*IC + ic] * input[batch * IC + ic];
    }
    output[batch*OC + block*block_length + t] = val;
  }
  // __syncthreads();
}

void LeNet5_cuda::predict(int batch) {
  // cpu_normalize(image, input);
  dim3 normGridDim(batch, 3, 1);
  dim3 normBlockDim(32, 32, 1);
  normalize<<<normGridDim, normBlockDim>>>(batch, input_channel, input_size, d_image, d_input);

  // dest, source, ...
  // cudaMemcpy(input, d_input, sizeof(float)*batch*input_channel*input_size*input_size, cudaMemcpyDeviceToHost);

  // Conv2d
  // cpu_conv(input, C1_feature_map, conv1_weight, conv1_bias, batch, input_size,
  //      input_size, conv1_in_channel, conv1_out_channel, conv1_kernel_size);
  // unrolled_conv(d_input, d_input_unrolled, d_C1_feature_map, d_conv1_weight, d_conv1_bias,
  //     batch, input_size, input_size, conv1_in_channel, conv1_out_channel, conv1_kernel_size);

  dim3 conv1GridDim(batch, 6, 1);
  dim3 conv1BlockDim(32, 32, 1);
  conv1<<<conv1GridDim, conv1BlockDim>>>(d_input, d_C1_feature_map, batch, input_size,
                                              input_size, conv1_in_channel, conv1_out_channel, conv1_kernel_size);

  // cudaMemcpy(C1_feature_map, d_C1_feature_map, sizeof(float)*batch*conv1_out_channel*C1_size*C1_size, cudaMemcpyDeviceToHost);

  // cpu_relu(C1_feature_map, batch * C1_channel * C1_size * C1_size);
  dim3 relu1GridDim(batch, 6, 1);
  dim3 relu1BlockDim(28, 28, 1);
  naive_relu<<<relu1GridDim, relu1BlockDim>>>(d_C1_feature_map, C1_channel, C1_size, C1_size);
  // cudaMemcpy(C1_feature_map, d_C1_feature_map, sizeof(float)*batch*conv1_out_channel*C1_size*C1_size, cudaMemcpyDeviceToHost);

  // MaxPool2d
  // cpu_pool(C1_feature_map, S2_feature_map, batch, C1_channel, C1_size, C1_size);
  dim3 pool1GridDim(batch, 6, 1);
  dim3 pool1BlockDim(14, 14, 1);
  naive_pool<<<pool1GridDim, pool1BlockDim>>>(d_C1_feature_map, d_S2_feature_map, C1_channel, C1_size, C1_size);
  // cudaMemcpy(S2_feature_map, d_S2_feature_map, sizeof(float)*batch*conv1_out_channel*S2_size*S2_size, cudaMemcpyDeviceToHost);

  // Conv2d
  // cpu_conv(S2_feature_map, C3_feature_map, conv2_weight, conv2_bias, batch, S2_size,
  //      S2_size, conv2_in_channel, conv2_out_channel, conv2_kernel_size);

  // unrolled_conv(d_S2_feature_map, d_S2_feature_map_unrolled, d_C3_feature_map, d_conv2_weight, d_conv2_bias,
  //     batch, S2_size, S2_size, conv2_in_channel, conv2_out_channel, conv2_kernel_size);

  dim3 conv2GridDim(batch, 16, 1);
  dim3 conv2BlockDim(14, 14, 1); // too few threads?
  conv2<<<conv2GridDim, conv2BlockDim>>>(d_S2_feature_map, d_C3_feature_map,
      batch, S2_size, S2_size, conv2_in_channel, conv2_out_channel, conv2_kernel_size);

  // cpu_relu(C3_feature_map, batch * C3_channel * C3_size * C3_size);
  dim3 relu2GridDim(batch, 16, 1);
  dim3 relu2BlockDim(10, 10, 1);
  naive_relu<<<relu2GridDim, relu2BlockDim>>>(d_C3_feature_map, C3_channel, C3_size, C3_size); 

  // MaxPool2d
  // cpu_pool(C3_feature_map, S4_feature_map, batch, C3_channel, C3_size, C3_size);
  dim3 pool2GridDim(batch, 16, 1);
  dim3 pool2BlockDim(5, 5, 1); // doesn't fill 32 threads per block -> underutilized
  naive_pool<<<pool2GridDim, pool2BlockDim>>>(d_C3_feature_map, d_S4_feature_map, C3_channel, C3_size, C3_size);
  // cudaMemcpy(S4_feature_map, d_S4_feature_map, sizeof(float)*batch*conv2_out_channel*S4_size*S4_size, cudaMemcpyDeviceToHost);

  // Linear
  // cpu_fc(S4_feature_map, C5_layer, fc1_weight, fc1_bias, batch, fc1_in_channel,
  //    fc1_out_channel);
  int fc1RowblockNum = 6;
  dim3 fc1GridDim(batch, fc1RowblockNum, 1);
  dim3 fc1BlockDim(512, 1, 1);
  fc_rowblock<<<fc1GridDim, fc1BlockDim, sizeof(float)*fc1_in_channel*(fc1_out_channel/fc1RowblockNum)>>>
          (d_S4_feature_map, d_C5_layer, d_fc1_weight, d_fc1_bias, 
      fc1_in_channel, fc1_out_channel, fc1RowblockNum);

  // cpu_relu(C5_layer, batch * C5_size);
  dim3 relu3GridDim(batch, 1, 1);
  dim3 relu3BlockDim(120, 1, 1);
  naive_relu<<<relu3GridDim, relu3BlockDim>>>(d_C5_layer, 1, 120, 1); 

  // Linear
  // cpu_fc(C5_layer, F6_layer, fc2_weight, fc2_bias, batch, fc2_in_channel,
  //    fc2_out_channel);
  int fc2RowblockNum = 7;
  dim3 fc2GridDim(batch, fc2RowblockNum, 1);
  dim3 fc2BlockDim(128, 1, 1);
  fc_rowblock<<<fc2GridDim, fc2BlockDim, sizeof(float)*fc2_in_channel*(fc2_out_channel/fc2RowblockNum)>>>
          (d_C5_layer, d_F6_layer, d_fc2_weight, d_fc2_bias, 
      fc2_in_channel, fc2_out_channel, fc2RowblockNum);

  // cpu_relu(F6_layer, batch * F6_size);
  dim3 relu4GridDim(batch, 1, 1);
  dim3 relu4BlockDim(84, 1, 1);
  naive_relu<<<relu4GridDim, relu4BlockDim>>>(d_F6_layer, 1, 84, 1); 

  // Linear
  // cpu_fc(F6_layer, output, fc3_weight, fc3_bias, batch, fc3_in_channel,
  //    fc3_out_channel);
  int fc3RowblockNum = 1;
  dim3 fc3GridDim(batch, fc3RowblockNum, 1);
  dim3 fc3BlockDim(128, 1, 1);
  fc_rowblock<<<fc3GridDim, fc3BlockDim, sizeof(float)*fc3_in_channel*(fc3_out_channel/fc3RowblockNum)>>>
          (d_F6_layer, d_output, d_fc3_weight, d_fc3_bias, 
      fc3_in_channel, fc3_out_channel, fc3RowblockNum);

  // dest, source, number of bytes, transfer type
  // cudaMemcpy(d_output, output, sizeof(float)*batch*output_size, cudaMemcpyHostToDevice);

  /* NOTE: unless you want to make a major change to this class structure, 
  *  you need to write your output to the device memory d_output 
  *  so that classify() can handle the rest.
  */
}

void LeNet5_cuda::prepare_device_memory(uint8_t* image) {
  // Store all double arrays as floats
  // Note: this was done here instead of loading everything as floats in LeNet5.cpp
  // in order to provide accuracy comparisons to the double CPU version.
  // No additional computations are performed.

  std::copy(this->conv1_weight, 
            this->conv1_weight+conv1_in_channel*conv1_out_channel*conv1_kernel_size*conv1_kernel_size,
            this->f_conv1_weight);
  // reorder_filters(this->conv1_weight, this->f_conv1_weight, conv1_in_channel, conv1_out_channel, conv1_kernel_size);
  std::copy(this->conv1_bias,
            this->conv1_bias+conv1_out_channel,
            this->f_conv1_bias);
  std::copy(this->conv2_weight,
            this->conv2_weight+conv2_in_channel*conv2_out_channel*conv2_kernel_size*conv2_kernel_size,
            this->f_conv2_weight);
  // reorder_filters(this->conv2_weight, this->f_conv2_weight, conv2_in_channel, conv2_out_channel, conv2_kernel_size);
  std::copy(this->conv2_bias,
            this->conv2_bias+conv2_out_channel,
            this->f_conv2_bias);
  std::copy(this->fc1_weight,
            this->fc1_weight+fc1_in_channel*fc1_out_channel,
            this->f_fc1_weight);
  std::copy(this->fc1_bias,
            this->fc1_bias+fc1_out_channel,
            this->f_fc1_bias);
  std::copy(this->fc2_weight,
            this->fc2_weight+fc2_in_channel*fc2_out_channel,
            this->f_fc2_weight);
  std::copy(this->fc2_bias,
            this->fc2_bias+fc2_out_channel,
            this->f_fc2_bias);
  std::copy(this->fc3_weight,
            this->fc3_weight+fc3_in_channel*fc3_out_channel,
            this->f_fc3_weight);
  std::copy(this->fc3_bias,
            this->fc3_bias+fc3_out_channel,
            this->f_fc3_bias);

  // For unrolled convolution
  // cudaMalloc((void**)&d_input_unrolled, sizeof(float)*
  //     batch * conv1_in_channel*conv1_kernel_size*conv1_kernel_size * C1_size*C1_size);
  // cudaMalloc((void**)&d_conv1_weight_unrolled, sizeof(float)*
  //     conv1_out_channel*conv1_kernel_size*conv1_kernel_size);
  // d_conv1_bias_unrolled; // for now just add scalar
  // cudaMalloc((void**)d_S2_feature_map_unrolled;
  // d_conv2_weight_unrolled;
  // d_conv2_bias_unrolled;
  // cudaMalloc((void**)&d_S2_feature_map_unrolled, sizeof(float)*
  //     batch * conv2_in_channel*conv2_kernel_size*conv2_kernel_size * S2_size*S2_size);
  // cudaMalloc((void**)&d_conv2_weight_unrolled, sizeof(float)*
  //     conv2_out_channel*conv2_kernel_size*conv2_kernel_size);

  // Alloc Model Parameters
  // cudaMalloc((void**)&d_conv1_weight,
  //            sizeof(float) * conv1_in_channel * conv1_out_channel *
  //                conv1_kernel_size * conv1_kernel_size);
  // cudaMalloc((void**)&d_conv1_bias, sizeof(float) * conv1_out_channel);
  // cudaMalloc((void**)&d_conv2_weight,
  //            sizeof(float) * conv2_in_channel * conv2_out_channel *
  //                conv2_kernel_size * conv2_kernel_size);
  // cudaMalloc((void**)&d_conv2_bias, sizeof(float) * conv2_out_channel);
  cudaMalloc((void**)&d_fc1_weight,
             sizeof(float) * fc1_in_channel * fc1_out_channel);
  cudaMalloc((void**)&d_fc1_bias, sizeof(float) * fc1_out_channel);
  cudaMalloc((void**)&d_fc2_weight,
             sizeof(float) * fc2_in_channel * fc2_out_channel);
  cudaMalloc((void**)&d_fc2_bias, sizeof(float) * fc2_out_channel);
  cudaMalloc((void**)&d_fc3_weight,
             sizeof(float) * fc3_in_channel * fc3_out_channel);
  cudaMalloc((void**)&d_fc3_bias, sizeof(float) * fc3_out_channel);

  // Alloc Activations
  cudaMalloc((void**)&d_image,
             sizeof(uint8_t) * batch * input_size * input_size * input_channel);
  cudaMalloc((void**)&d_input,
             sizeof(float) * batch * input_channel * input_size * input_size);
  cudaMalloc((void**)&d_C1_feature_map,
             sizeof(float) * batch * C1_channel * C1_size * C1_size);
  cudaMalloc((void**)&d_S2_feature_map,
             sizeof(float) * batch * S2_channel * S2_size * S2_size);
  cudaMalloc((void**)&d_C3_feature_map,
             sizeof(float) * batch * C3_channel * C3_size * C3_size);
  cudaMalloc((void**)&d_S4_feature_map,
             sizeof(float) * batch * S4_channel * S4_size * S4_size);
  cudaMalloc((void**)&d_C5_layer, sizeof(float) * batch * C5_size);
  cudaMalloc((void**)&d_F6_layer, sizeof(float) * batch * F6_size);
  cudaMalloc((void**)&d_output, sizeof(float) * batch * output_size);

  // Copy Parameters
  cudaMemcpyToSymbol(d_conv1_weight, f_conv1_weight,
             sizeof(float) * conv1_in_channel * conv1_out_channel *
                 conv1_kernel_size * conv1_kernel_size);
  cudaMemcpyToSymbol(d_conv1_bias, f_conv1_bias, sizeof(float) * conv1_out_channel);
  cudaMemcpyToSymbol(d_conv2_weight, f_conv2_weight,
             sizeof(float) * conv2_in_channel * conv2_out_channel *
                 conv2_kernel_size * conv2_kernel_size);
  cudaMemcpyToSymbol(d_conv2_bias, f_conv2_bias, sizeof(float) * conv2_out_channel);

  cudaMemcpy(d_fc1_weight, f_fc1_weight,
             sizeof(float) * fc1_in_channel * fc1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc1_bias, f_fc1_bias, sizeof(float) * fc1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc2_weight, f_fc2_weight,
             sizeof(float) * fc2_in_channel * fc2_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc2_bias, f_fc2_bias, sizeof(float) * fc2_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc3_weight, f_fc3_weight,
             sizeof(float) * fc3_in_channel * fc3_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc3_bias, f_fc3_bias, sizeof(float) * fc3_out_channel,
             cudaMemcpyHostToDevice);

  // copy input image
  size_t image_size = batch * input_size * input_size * input_channel;
  cudaMemcpy(d_image, image, image_size * sizeof(uint8_t),
             cudaMemcpyHostToDevice);
}

void LeNet5_cuda::classify(int* predict, int batch) {
  // read logits back to cpu
  cudaMemcpy(f_output, d_output, sizeof(float) * output_size * batch,
             cudaMemcpyDeviceToHost);

  // float back to double
  std::copy(f_output,
            f_output+batch*output_size,
            output);
  // Softmax
  softmax(output, predict, batch, output_size);
}

LeNet5_cuda::~LeNet5_cuda() {
  cudaFree(d_conv1_weight);   
  cudaFree(d_conv2_weight);   
  cudaFree(d_conv1_bias);     
  cudaFree(d_conv2_bias);     
  cudaFree(d_fc1_weight);     
  cudaFree(d_fc2_weight);     
  cudaFree(d_fc3_weight);     
  cudaFree(d_fc1_bias);       
  cudaFree(d_fc2_bias);       
  cudaFree(d_fc3_bias);       

  cudaFree(d_image);          
  cudaFree(d_input);          
  cudaFree(d_C1_feature_map); 
  cudaFree(d_S2_feature_map); 
  cudaFree(d_C3_feature_map); 
  cudaFree(d_S4_feature_map); 
  cudaFree(d_C5_layer);      
  cudaFree(d_F6_layer);     
  cudaFree(d_output);       
  cudaFree(d_predict_cuda);   
  
  // Free model parameters memories
  delete[] this->f_conv1_weight;
  delete[] this->f_conv1_bias;
  delete[] this->f_conv2_weight;
  delete[] this->f_conv2_bias;
  delete[] this->f_fc1_weight;
  delete[] this->f_fc1_bias;
  delete[] this->f_fc2_weight;
  delete[] this->f_fc2_bias;
  delete[] this->f_fc3_weight;
  delete[] this->f_fc3_bias;
  // // Free activation memories
  // delete[] this->f_input;
  // delete[] this->f_C1_feature_map;
  // delete[] this->f_S2_feature_map;
  // delete[] this->f_C3_feature_map;
  // delete[] this->f_S4_feature_map;
  // delete[] this->f_C5_layer;
  // delete[] this->f_F6_layer;
  delete[] this->f_output;

  // free unrolled
  cudaFree(d_input_unrolled);
  cudaFree(d_conv1_weight_unrolled);
  // float* d_conv1_bias_unrolled; // is this necessary?
  cudaFree(d_S2_feature_map_unrolled);
  cudaFree(d_conv2_weight_unrolled);
  // float* d_conv2_bias_unrolled;
}


/*** CPU fallbacks ***/

void LeNet5_cuda::cpu_normalize(const uint8_t* const image, float* input) {
  // Initialize variables
  float max_int = 255.0L;
  float mean = 0.5L;
  float var = 0.5L;
  // Normalize
  for (int i = 0; i < batch * input_channel * input_size * input_size; i++) {
    input[i] = image[i] / max_int;       // transforms.ToTensor();
    input[i] = (input[i] - mean) / var;  // transforms.Normalize();
  }
}

void LeNet5_cuda::cpu_relu(float* feature_map, int size) {
  // relu
  for (int i = 0; i < size; i++) feature_map[i] = std::max(feature_map[i], 0.0f);
}

void LeNet5_cuda::cpu_conv(float* input, float* output, float* weight,
                      float* bias, int B, int H, int W, int IC, int OC,
                      int K) {
  // Initialize variable
  int H_OUT = H - (K - 1);
  int W_OUT = W - (K - 1);
  // Convolution
  for (int b = 0; b < B; b++)              // mini-batch
    for (int oc = 0; oc < OC; oc++) {      // Output Channel
      for (int h = 0; h < H_OUT; h++)      // Height
        for (int w = 0; w < W_OUT; w++) {  // Width
          int output_index =
              b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * W_OUT + w;
          output[output_index] = bias[oc];
          for (int ic = 0; ic < IC; ic++) {
            int input_base = b * (IC * H * W) + ic * (H * W) + h * (W) + w;
            int kernel_base = oc * (IC * K * K) + ic * (K * K);
            for (int kh = 0; kh < K; kh++)
              for (int kw = 0; kw < K; kw++) {
                float val = input[input_base + kh * (W) + kw] *
                             weight[kernel_base + kh * (K) + kw];
                output[output_index] += val;
              }
          }
        }
    }
}

void LeNet5_cuda::cpu_pool(float* input, float* output, int B, int C, int H,
                      int W) {
  // Initilaize variable
  int scale = 2;
  int H_OUT = H / scale;
  int W_OUT = W / scale;
  // Max Pooling
  for (int b = 0; b < B; b++)
    for (int c = 0; c < C; c++)
      for (int h = 0; h < H; h += 2)
        for (int w = 0; w < W; w += 2) {
          // Init values
          int input_base = b * (C * H * W) + c * (H * W) + h * (W) + w;
          int max_sh = 0;
          int max_sw = 0;
          float max_val = std::numeric_limits<float>::lowest();
          // Find maximum
          for (int sh = 0; sh < scale; sh++)
            for (int sw = 0; sw < scale; sw++) {
              float val = input[input_base + sh * (W) + sw];
              if (val - max_val > std::numeric_limits<float>::epsilon()) {
                max_val = val;
                max_sh = sh;
                max_sw = sw;
              }
            }
          // Set output with max value
          int output_index = b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) +
                             (h / 2) * W_OUT + (w / 2);
          output[output_index] = max_val;
        }
}

void LeNet5_cuda::cpu_fc(float* input, float* output, float* weight, float* bias,
                    int B, int IC, int OC) {
  // Fully Connected
  for (int b = 0; b < B; b++)
    for (int oc = 0; oc < OC; oc++) {
      output[b * OC + oc] = bias[oc];
      for (int ic = 0; ic < IC; ic++)
        output[b * OC + oc] += weight[oc * IC + ic] * input[b * IC + ic];
    }
}
