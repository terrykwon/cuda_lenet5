#include "LeNet5.h"

LeNet5::LeNet5(int batch) {
  // Internal variable
  this->batch = batch;
  // Model Parameters
  this->conv1_weight = new double[conv1_in_channel * conv1_out_channel *
                                  conv1_kernel_size * conv1_kernel_size];
  this->conv1_bias = new double[conv1_out_channel];
  this->conv2_weight = new double[conv2_in_channel * conv2_out_channel *
                                  conv2_kernel_size * conv2_kernel_size];
  this->conv2_bias = new double[conv2_out_channel];
  this->fc1_weight = new double[fc1_in_channel * fc1_out_channel];
  this->fc1_bias = new double[fc1_out_channel];
  this->fc2_weight = new double[fc2_in_channel * fc2_out_channel];
  this->fc2_bias = new double[fc2_out_channel];
  this->fc3_weight = new double[fc3_in_channel * fc3_out_channel];
  this->fc3_bias = new double[fc3_out_channel];
  // Activation
  this->input = new double[batch * input_channel * input_size * input_size];
  this->C1_feature_map = new double[batch * C1_channel * C1_size * C1_size];
  this->S2_feature_map = new double[batch * S2_channel * S2_size * S2_size];
  this->C3_feature_map = new double[batch * C3_channel * C3_size * C3_size];
  this->S4_feature_map = new double[batch * S4_channel * S4_size * S4_size];
  this->C5_layer = new double[batch * C5_size];
  this->F6_layer = new double[batch * F6_size];
  this->output = new double[batch * output_size];
}

LeNet5::~LeNet5() {
  // Free model parameters memories
  delete[] this->conv1_weight;
  delete[] this->conv1_bias;
  delete[] this->conv2_weight;
  delete[] this->conv2_bias;
  delete[] this->fc1_weight;
  delete[] this->fc1_bias;
  delete[] this->fc2_weight;
  delete[] this->fc2_bias;
  delete[] this->fc3_weight;
  delete[] this->fc3_bias;
  // Free activation memories
  delete[] this->input;
  delete[] this->C1_feature_map;
  delete[] this->S2_feature_map;
  delete[] this->C3_feature_map;
  delete[] this->S4_feature_map;
  delete[] this->C5_layer;
  delete[] this->F6_layer;
  delete[] this->output;
}

void LeNet5::load_parameters(std::string value_path) {
  // Load parameters from value_path
  {
    // Initialize variables
    std::string buffer;
    std::ifstream value_file;
    // Open file
    value_file.open(value_path);
    // conv1.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int c = 0; c < conv1_in_channel * conv1_out_channel; c++) {
      for (int i = 0; i < conv1_kernel_size; i++)
        for (int j = 0; j < conv1_kernel_size; j++)
          value_file >>
              conv1_weight[c * (conv1_kernel_size * conv1_kernel_size) +
                           i * conv1_kernel_size + j];
    }
    // conv2.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int c = 0; c < conv2_in_channel * conv2_out_channel; c++) {
      for (int i = 0; i < conv2_kernel_size; i++)
        for (int j = 0; j < conv2_kernel_size; j++)
          value_file >>
              conv2_weight[c * (conv2_kernel_size * conv2_kernel_size) +
                           i * conv2_kernel_size + j];
    }
    // conv1.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < conv1_out_channel; oc++) value_file >> conv1_bias[oc];
    // conv2.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < conv2_out_channel; oc++) value_file >> conv2_bias[oc];
    // fc1.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < fc1_out_channel; oc++) {
      for (int ic = 0; ic < fc1_in_channel; ic++) {
        value_file >> fc1_weight[oc * fc1_in_channel + ic];
      }
    }
    // fc2.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < fc2_out_channel; oc++) {
      for (int ic = 0; ic < fc2_in_channel; ic++) {
        value_file >> fc2_weight[oc * fc2_in_channel + ic];
      }
    }
    // fc3.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < fc3_out_channel; oc++) {
      for (int ic = 0; ic < fc3_in_channel; ic++) {
        value_file >> fc3_weight[oc * fc3_in_channel + ic];
      }
    }
    // fc1.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < fc1_out_channel; oc++) {
      value_file >> fc1_bias[oc];
    }
    // fc2.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < fc2_out_channel; oc++) {
      value_file >> fc2_bias[oc];
    }
    // fc3.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < fc3_out_channel; oc++) {
      value_file >> fc3_bias[oc];
    }
    // Close file
    value_file.close();
  }
}

void LeNet5::print_parameters() {
  std::cout.precision(std::numeric_limits<double>::max_digits10);
  // conv1.weight
  for (int c = 0; c < conv1_in_channel * conv1_out_channel; c++) {
    std::cout << "conv1.weight.c" << c + 1 << std::endl;
    for (int i = 0; i < conv1_kernel_size; i++) {
      for (int j = 0; j < conv1_kernel_size; j++) {
        std::cout << conv1_weight[c * (conv1_kernel_size * conv1_kernel_size) +
                                  i * conv1_kernel_size + j]
                  << " ";
      }
      std::cout << std::endl;
    }
  }
  // conv2.weight
  for (int c = 0; c < conv2_in_channel * conv2_out_channel; c++) {
    std::cout << "conv2.weight.c" << c + 1 << std::endl;
    for (int i = 0; i < conv2_kernel_size; i++) {
      for (int j = 0; j < conv2_kernel_size; j++) {
        std::cout << conv2_weight[c * (conv2_kernel_size * conv2_kernel_size) +
                                  i * conv2_kernel_size + j]
                  << " ";
      }
      std::cout << std::endl;
    }
  }
  // conv1.bias
  std::cout << "conv1.bias" << std::endl;
  for (int oc = 0; oc < conv1_out_channel; oc++) {
    std::cout << conv1_bias[oc] << " ";
  }
  std::cout << std::endl;
  // conv2.bias
  std::cout << "conv2.bias" << std::endl;
  for (int oc = 0; oc < conv2_out_channel; oc++) {
    std::cout << conv2_bias[oc] << " ";
  }
  std::cout << std::endl;
  // fc1.weight
  for (int oc = 0; oc < fc1_out_channel; oc++) {
    std::cout << "fc1.weight.out_channel" << oc + 1 << std::endl;
    for (int ic = 0; ic < fc1_in_channel; ic++) {
      std::cout << fc1_weight[oc * fc1_in_channel + ic] << " ";
    }
    std::cout << std::endl;
  }
  // fc2.weight
  for (int oc = 0; oc < fc2_out_channel; oc++) {
    std::cout << "fc2.weight.out_channel" << oc + 1 << std::endl;
    for (int ic = 0; ic < fc2_in_channel; ic++) {
      std::cout << fc2_weight[oc * fc2_in_channel + ic] << " ";
    }
    std::cout << std::endl;
  }
  // fc3.weight
  for (int oc = 0; oc < fc3_out_channel; oc++) {
    std::cout << "fc3.weight.out_channel" << oc + 1 << std::endl;
    for (int ic = 0; ic < fc3_in_channel; ic++) {
      std::cout << fc3_weight[oc * fc3_in_channel + ic] << " ";
    }
    std::cout << std::endl;
  }
  // fc1.bias
  std::cout << "fc1.bias" << std::endl;
  for (int oc = 0; oc < fc1_out_channel; oc++) {
    std::cout << fc1_bias[oc] << " ";
  }
  std::cout << std::endl;
  // fc2.bias
  std::cout << "fc2.bias" << std::endl;
  for (int oc = 0; oc < fc2_out_channel; oc++) {
    std::cout << fc2_bias[oc] << " ";
  }
  std::cout << std::endl;
  // fc3.bias
  std::cout << "fc3.bias" << std::endl;
  for (int oc = 0; oc < fc3_out_channel; oc++) {
    std::cout << fc3_bias[oc] << " ";
  }
  std::cout << std::endl;
}

void LeNet5::softmax(double* input, int* output, int B, int size) {
  for (int b = 0; b < B; b++) {
    // Initialize
    int max_idx = 0;
    double max_val = std::exp(std::numeric_limits<double>::lowest());
    // calcualte Z = sum_all(exp(x_i))
    double Z = 0;
    for (int i = 0; i < size; i++) Z += std::exp(input[b * size + i]);
    // Softmax
    for (int i = 0; i < size; i++) {
      input[b * size + i] = std::exp(input[b * size + i]) / Z;
      if (input[i] - max_val > std::numeric_limits<double>::epsilon()) {
        max_val = input[b * size + i];
        max_idx = i;
      }
    }
    output[b] = max_idx;
  }
}

bool LeNet5::compare(LeNet5* other) {
  // TODO: Implement this...
  return true;
}
