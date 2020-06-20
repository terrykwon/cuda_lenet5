#include "LeNet5_cuda.h"


/**
 * Tiled matrix multiplication code from textbook
 */
// __global__
// void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width) {
//   __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
//   __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

//   int bx = blockIdx.x;
//   int by = blockIdx.y;
//   int tx = threadIdx.x;
//   int ty = threadIdx.y;

//   int Row = by * TILE_WIDTH + ty;
//   int Col = bx * TILE_WIDTH + tx;

//   float Pvalue = 0;

//   for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
//     // collaborative loading of d_M and d_N tiles into shared memory
//     // important! should understand
//     Mds[ty][tx] = d_M[Row*Width + ph*TILE_WIDTH * tx];
//     Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty)*Width + Col];
//     __syncthreads();

//     for (int k = 0; k < TILE_WIDTH; k++) {
//       Pvalue += Mds[ty][k] * Nds[k][tx];
//     }
//     __syncthreads(); // why?
//   }
//   d_P[Row*Width + Col] = Pvalue;
// }

/**
 * (batch_size, 3, 1) x (32, 32, 1)
 */
__global__ void normalize(int batch, int input_channel, int input_size, const uint8_t* const d_image, double* d_input) {
  // automatically placed in registers. Should these be in shared / constant?
  // probably not because they're just single variables
  const double max_int = 255.0L;   
  const double mean = 0.5L;
  const double var = 0.5L;

  const int batch_id = blockIdx.x;
  const int channel_id = blockIdx.y;
  const int col = threadIdx.x;
  const int row = threadIdx.y;

  double val;

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
__global__ void naive_conv(double* input, double* output, double* weight,
                      double* bias, int B, int H, int W, int IC, int OC,
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
        double val = input[input_base + kh * (W) + kw] *
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
__global__ void naive_relu(double* feature_map, int channels, int width, int height) {
  int b = blockIdx.x; // batch
  int oc = blockIdx.y; // output channel
  int w = threadIdx.x; // col
  int h = threadIdx.y; // row

  int index = b * (channels * width * height) + oc * (width * height) + h * width + w;

  feature_map[index] = fmax(feature_map[index], 0.0);
}

/**
 * (batch_size, in_channels, 1) x (width, height, 1)
 * This is actually much slower than the CPU version.
 */
__global__ void naive_pool(double* input, double* output, int C, int H, int W) {
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
  double max_val = 0.0; // since after relu

  // Find maximum
  for (int sh = 0; sh < scale; sh++) {
    for (int sw = 0; sw < scale; sw++) {
      double val = input[input_base + sh * (W) + sw];
      if (val > max_val) {
        max_val = val;
        // max_sh = sh;
        // max_sw = sw;
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
__global__ void naive_fc(double* input, double* output, double* weight, double* bias,
                         int IC, int OC) {
  int b = blockIdx.x;
  int oc = threadIdx.x; 

  output[b * OC + oc] = bias[oc];
  for (int ic = 0; ic < IC; ic++)
    output[b * OC + oc] += weight[oc * IC + ic] * input[b * IC + ic];
}

void LeNet5_cuda::predict(int batch) {
  std::cout << "batch " << batch << std::endl;

  // cpu_normalize(image, input);
  dim3 normGridDim(batch, 3, 1);
  dim3 normBlockDim(32, 32, 1);
  normalize<<<normGridDim, normBlockDim>>>(batch, input_channel, input_size, d_image, d_input);

  // dest, source, ...
  // cudaMemcpy(input, d_input, sizeof(double)*batch*input_channel*input_size*input_size, cudaMemcpyDeviceToHost);

  // Conv2d
  // cpu_conv(input, C1_feature_map, conv1_weight, conv1_bias, batch, input_size,
  //      input_size, conv1_in_channel, conv1_out_channel, conv1_kernel_size);
  dim3 conv1GridDim(batch, 6, 1);
  dim3 conv1BlockDim(28, 28, 1);
  naive_conv<<<conv1GridDim, conv1BlockDim>>>(d_input, d_C1_feature_map, d_conv1_weight, d_conv1_bias, batch, input_size,
                                              input_size, conv1_in_channel, conv1_out_channel, conv1_kernel_size);

  // cudaMemcpy(C1_feature_map, d_C1_feature_map, sizeof(double)*batch*conv1_out_channel*C1_size*C1_size, cudaMemcpyDeviceToHost);

  // cpu_relu(C1_feature_map, batch * C1_channel * C1_size * C1_size);
  dim3 relu1GridDim(batch, 6, 1);
  dim3 relu1BlockDim(28, 28, 1);
  naive_relu<<<relu1GridDim, relu1BlockDim>>>(d_C1_feature_map, C1_channel, C1_size, C1_size);
  // cudaMemcpy(C1_feature_map, d_C1_feature_map, sizeof(double)*batch*conv1_out_channel*C1_size*C1_size, cudaMemcpyDeviceToHost);

  // MaxPool2d
  // cpu_pool(C1_feature_map, S2_feature_map, batch, C1_channel, C1_size, C1_size);
  dim3 pool1GridDim(batch, 6, 1);
  dim3 pool1BlockDim(14, 14, 1);
  naive_pool<<<pool1GridDim, pool1BlockDim>>>(d_C1_feature_map, d_S2_feature_map, C1_channel, C1_size, C1_size);
  // cudaMemcpy(S2_feature_map, d_S2_feature_map, sizeof(double)*batch*conv1_out_channel*S2_size*S2_size, cudaMemcpyDeviceToHost);

  // Conv2d
  // cpu_conv(S2_feature_map, C3_feature_map, conv2_weight, conv2_bias, batch, S2_size,
  //      S2_size, conv2_in_channel, conv2_out_channel, conv2_kernel_size);
  dim3 conv2GridDim(batch, 16, 1);
  dim3 conv2BlockDim(10, 10, 1); // too few threads?
  naive_conv<<<conv2GridDim, conv2BlockDim>>>(d_S2_feature_map, d_C3_feature_map, d_conv2_weight, 
      d_conv2_bias, batch, S2_size, S2_size, conv2_in_channel, conv2_out_channel, conv2_kernel_size);

  // cpu_relu(C3_feature_map, batch * C3_channel * C3_size * C3_size);
  dim3 relu2GridDim(batch, 16, 1);
  dim3 relu2BlockDim(10, 10, 1);
  naive_relu<<<relu2GridDim, relu2BlockDim>>>(d_C3_feature_map, C3_channel, C3_size, C3_size); 

  // MaxPool2d
  // cpu_pool(C3_feature_map, S4_feature_map, batch, C3_channel, C3_size, C3_size);
  dim3 pool2GridDim(batch, 16, 1);
  dim3 pool2BlockDim(5, 5, 1);
  naive_pool<<<pool2GridDim, pool2BlockDim>>>(d_C3_feature_map, d_S4_feature_map, C3_channel, C3_size, C3_size);
  // cudaMemcpy(S4_feature_map, d_S4_feature_map, sizeof(double)*batch*conv2_out_channel*S4_size*S4_size, cudaMemcpyDeviceToHost);

  // Linear
  // cpu_fc(S4_feature_map, C5_layer, fc1_weight, fc1_bias, batch, fc1_in_channel,
  //    fc1_out_channel);
  dim3 fc1GridDim(batch, 1, 1);
  dim3 fc1BlockDim(120, 1, 1);
  naive_fc<<<fc1GridDim, fc1BlockDim>>>(d_S4_feature_map, d_C5_layer, d_fc1_weight, d_fc1_bias, 
      fc1_in_channel, fc1_out_channel);

  // cpu_relu(C5_layer, batch * C5_size);
  dim3 relu3GridDim(batch, 1, 1);
  dim3 relu3BlockDim(120, 1, 1);
  naive_relu<<<relu3GridDim, relu3BlockDim>>>(d_C5_layer, 1, 120, 1); 

  // Linear
  // cpu_fc(C5_layer, F6_layer, fc2_weight, fc2_bias, batch, fc2_in_channel,
  //    fc2_out_channel);
  dim3 fc2GridDim(batch, 1, 1);
  dim3 fc2BlockDim(84, 1, 1);
  naive_fc<<<fc2GridDim, fc2BlockDim>>>(d_C5_layer, d_F6_layer, d_fc2_weight, d_fc2_bias, 
      fc2_in_channel, fc2_out_channel);

  // cpu_relu(F6_layer, batch * F6_size);
  dim3 relu4GridDim(batch, 1, 1);
  dim3 relu4BlockDim(84, 1, 1);
  naive_relu<<<relu4GridDim, relu4BlockDim>>>(d_F6_layer, 1, 84, 1); 

  // Linear
  // cpu_fc(F6_layer, output, fc3_weight, fc3_bias, batch, fc3_in_channel,
  //    fc3_out_channel);
  dim3 fc3GridDim(batch, 1, 1);
  dim3 fc3BlockDim(10, 1, 1);
  naive_fc<<<fc3GridDim, fc3BlockDim>>>(d_F6_layer, d_output, d_fc3_weight, d_fc3_bias, 
      fc3_in_channel, fc3_out_channel);

  // dest, source, number of bytes, transfer type
  // cudaMemcpy(d_output, output, sizeof(double)*batch*output_size, cudaMemcpyHostToDevice);

  /* NOTE: unless you want to make a major change to this class structure, 
  *  you need to write your output to the device memory d_output 
  *  so that classify() can handle the rest.
  */
}

void LeNet5_cuda::prepare_device_memory(uint8_t* image) {
  // Alloc Model Parameters
  cudaMalloc((void**)&d_conv1_weight,
             sizeof(double) * conv1_in_channel * conv1_out_channel *
                 conv1_kernel_size * conv1_kernel_size);
  cudaMalloc((void**)&d_conv1_bias, sizeof(double) * conv1_out_channel);
  cudaMalloc((void**)&d_conv2_weight,
             sizeof(double) * conv2_in_channel * conv2_out_channel *
                 conv2_kernel_size * conv2_kernel_size);
  cudaMalloc((void**)&d_conv2_bias, sizeof(double) * conv2_out_channel);
  cudaMalloc((void**)&d_fc1_weight,
             sizeof(double) * fc1_in_channel * fc1_out_channel);
  cudaMalloc((void**)&d_fc1_bias, sizeof(double) * fc1_out_channel);
  cudaMalloc((void**)&d_fc2_weight,
             sizeof(double) * fc2_in_channel * fc2_out_channel);
  cudaMalloc((void**)&d_fc2_bias, sizeof(double) * fc2_out_channel);
  cudaMalloc((void**)&d_fc3_weight,
             sizeof(double) * fc3_in_channel * fc3_out_channel);
  cudaMalloc((void**)&d_fc3_bias, sizeof(double) * fc3_out_channel);

  // Alloc Activations
  cudaMalloc((void**)&d_image,
             sizeof(uint8_t) * batch * input_size * input_size * input_channel);
  cudaMalloc((void**)&d_input,
             sizeof(double) * batch * input_channel * input_size * input_size);
  cudaMalloc((void**)&d_C1_feature_map,
             sizeof(double) * batch * C1_channel * C1_size * C1_size);
  cudaMalloc((void**)&d_S2_feature_map,
             sizeof(double) * batch * S2_channel * S2_size * S2_size);
  cudaMalloc((void**)&d_C3_feature_map,
             sizeof(double) * batch * C3_channel * C3_size * C3_size);
  cudaMalloc((void**)&d_S4_feature_map,
             sizeof(double) * batch * S4_channel * S4_size * S4_size);
  cudaMalloc((void**)&d_C5_layer, sizeof(double) * batch * C5_size);
  cudaMalloc((void**)&d_F6_layer, sizeof(double) * batch * F6_size);
  cudaMalloc((void**)&d_output, sizeof(double) * batch * output_size);

  // Copy Parameters
  cudaMemcpy(d_conv1_weight, conv1_weight,
             sizeof(double) * conv1_in_channel * conv1_out_channel *
                 conv1_kernel_size * conv1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv1_bias, conv1_bias, sizeof(double) * conv1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_weight, conv2_weight,
             sizeof(double) * conv2_in_channel * conv2_out_channel *
                 conv2_kernel_size * conv2_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_bias, conv2_bias, sizeof(double) * conv2_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc1_weight, fc1_weight,
             sizeof(double) * fc1_in_channel * fc1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc1_bias, fc1_bias, sizeof(double) * fc1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc2_weight, fc2_weight,
             sizeof(double) * fc2_in_channel * fc2_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc2_bias, fc2_bias, sizeof(double) * fc2_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc3_weight, fc3_weight,
             sizeof(double) * fc3_in_channel * fc3_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc3_bias, fc3_bias, sizeof(double) * fc3_out_channel,
             cudaMemcpyHostToDevice);
  // copy input image
  size_t image_size = batch * input_size * input_size * input_channel;
  cudaMemcpy(d_image, image, image_size * sizeof(uint8_t),
             cudaMemcpyHostToDevice);
}

void LeNet5_cuda::classify(int* predict, int batch) {
  // read logits back to cpu
  cudaMemcpy(output, d_output, sizeof(double) * output_size * batch,
             cudaMemcpyDeviceToHost);
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
}


/*** CPU fallbacks ***/

void LeNet5_cuda::cpu_normalize(const uint8_t* const image, double* input) {
  // Initialize variables
  double max_int = 255.0L;
  double mean = 0.5L;
  double var = 0.5L;
  // Normalize
  for (int i = 0; i < batch * input_channel * input_size * input_size; i++) {
    input[i] = image[i] / max_int;       // transforms.ToTensor();
    input[i] = (input[i] - mean) / var;  // transforms.Normalize();
  }
}

void LeNet5_cuda::cpu_relu(double* feature_map, int size) {
  // relu
  for (int i = 0; i < size; i++) feature_map[i] = std::max(feature_map[i], 0.0);
}

void LeNet5_cuda::cpu_conv(double* input, double* output, double* weight,
                      double* bias, int B, int H, int W, int IC, int OC,
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
                double val = input[input_base + kh * (W) + kw] *
                             weight[kernel_base + kh * (K) + kw];
                output[output_index] += val;
              }
          }
        }
    }
}

void LeNet5_cuda::cpu_pool(double* input, double* output, int B, int C, int H,
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
          double max_val = std::numeric_limits<double>::lowest();
          // Find maximum
          for (int sh = 0; sh < scale; sh++)
            for (int sw = 0; sw < scale; sw++) {
              double val = input[input_base + sh * (W) + sw];
              if (val - max_val > std::numeric_limits<double>::epsilon()) {
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

void LeNet5_cuda::cpu_fc(double* input, double* output, double* weight, double* bias,
                    int B, int IC, int OC) {
  // Fully Connected
  for (int b = 0; b < B; b++)
    for (int oc = 0; oc < OC; oc++) {
      output[b * OC + oc] = bias[oc];
      for (int ic = 0; ic < IC; ic++)
        output[b * OC + oc] += weight[oc * IC + ic] * input[b * IC + ic];
    }
}