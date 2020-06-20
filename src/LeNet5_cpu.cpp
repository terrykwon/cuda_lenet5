#include "LeNet5_cpu.h"

void LeNet5_cpu::predict(const uint8_t* const image, int batch) {
  // ToTensor and Normalize
  normalize(image, input);
  // Conv2d
  conv(input, C1_feature_map, conv1_weight, conv1_bias, batch, input_size,
       input_size, conv1_in_channel, conv1_out_channel, conv1_kernel_size);
  relu(C1_feature_map, batch * C1_channel * C1_size * C1_size);
  // MaxPool2d
  pool(C1_feature_map, S2_feature_map, batch, C1_channel, C1_size, C1_size);
  // Conv2d
  conv(S2_feature_map, C3_feature_map, conv2_weight, conv2_bias, batch, S2_size,
       S2_size, conv2_in_channel, conv2_out_channel, conv2_kernel_size);
  relu(C3_feature_map, batch * C3_channel * C3_size * C3_size);
  // MaxPool2d
  pool(C3_feature_map, S4_feature_map, batch, C3_channel, C3_size, C3_size);
  // Linear
  fc(S4_feature_map, C5_layer, fc1_weight, fc1_bias, batch, fc1_in_channel,
     fc1_out_channel);
  relu(C5_layer, batch * C5_size);
  // Linear
  fc(C5_layer, F6_layer, fc2_weight, fc2_bias, batch, fc2_in_channel,
     fc2_out_channel);
  relu(F6_layer, batch * F6_size);
  // Linear
  fc(F6_layer, output, fc3_weight, fc3_bias, batch, fc3_in_channel,
     fc3_out_channel);
}

void LeNet5_cpu::normalize(const uint8_t* const image, double* input) {
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

void LeNet5_cpu::relu(double* feature_map, int size) {
  // relu
  for (int i = 0; i < size; i++) feature_map[i] = std::max(feature_map[i], 0.0);
}

void LeNet5_cpu::conv(double* input, double* output, double* weight,
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

void LeNet5_cpu::pool(double* input, double* output, int B, int C, int H,
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

void LeNet5_cpu::fc(double* input, double* output, double* weight, double* bias,
                    int B, int IC, int OC) {
  // Fully Connected
  for (int b = 0; b < B; b++)
    for (int oc = 0; oc < OC; oc++) {
      output[b * OC + oc] = bias[oc];
      for (int ic = 0; ic < IC; ic++)
        output[b * OC + oc] += weight[oc * IC + ic] * input[b * IC + ic];
    }
}

void LeNet5_cpu::classify(int* predict, int batch) {
  // Softmax
  softmax(output, predict, batch, output_size);
}

void LeNet5_cpu::print_fc(double* data, int size) {
  printf("[DEBUG] print %d\n", size);
  for (int i = 0; i < size; i++) printf("%lf\n", data[i]);
}

void LeNet5_cpu::print_C1() {
  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < C1_channel; c++) {
      for (int h = 0; h < C1_size; h++) {
        for (int w = 0; w < C1_size; w++) {
          printf("%lf ",
                 C1_feature_map[b * (C1_channel * C1_size * C1_size) +
                                c * (C1_size * C1_size) + h * (C1_size) + w]);
        }
        printf("\n");
      }
      printf("\n");
    }
  }
}

void LeNet5_cpu::print_C3() {
  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < C3_channel; c++) {
      for (int h = 0; h < C3_size; h++) {
        for (int w = 0; w < C3_size; w++) {
          printf("%lf ",
                 C3_feature_map[b * (C3_channel * C3_size * C3_size) +
                                c * (C3_size * C3_size) + h * (C3_size) + w]);
        }
        printf("\n");
      }
      printf("\n");
    }
  }
}
