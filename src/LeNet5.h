#ifndef LENET5_H
#define LENET5_H

#include "common.h"
#include <limits>
#include <cmath>
#include <algorithm>

class LeNet5
{
public:
    LeNet5(int batch = 1);
    ~LeNet5();
    virtual void load_parameters(std::string value_path);
    virtual void print_parameters();
    virtual bool compare(LeNet5* other);
    virtual void predict(const uint8_t* const image, int batch) = 0;
    virtual void classify(int* predict, int batch) = 0;
protected:
    void softmax(double* input, int* output, int B, int size);
    //////////////////////////////////////////////////
    // Internal parameter
    //////////////////////////////////////////////////
    int batch = 1;
    int parameter_initialized = false;
    //////////////////////////////////////////////////
    // Model Parameters
    //////////////////////////////////////////////////
    double* conv1_weight;   // [3][6][5][5];
    double* conv2_weight;   // [6][16][5][5];
    double* conv1_bias;     // [6];
    double* conv2_bias;     // [16];
    double* fc1_weight;     // [400][120];
    double* fc2_weight;     // [120][84];
    double* fc3_weight;     // [84][10];
    double* fc1_bias;       // [120];
    double* fc2_bias;       // [84];
    double* fc3_bias;       // [10];
    //////////////////////////////////////////////////
    // Feature Map
    //////////////////////////////////////////////////
    double* input;          // [batch][3][32][32];
    double* C1_feature_map; // [batch][6][28][28];
    double* S2_feature_map; // [batch][6][14][14];
    double* C3_feature_map; // [batch][16][10][10];
    double* S4_feature_map; // [batch][16][5][5];
    double* C5_layer;       // [batch][120];
    double* F6_layer;       // [batch][84];
    double* output;         // [batch][10];
    //////////////////////////////////////////////////
    // Layer and Feature map parameters
    //     Check README.md for more information
    //////////////////////////////////////////////////
    //// Input
    int input_size = 32;
    int input_channel = 3;
    //// [Layer] conv1
    int conv1_in_channel = 3;
    int conv1_out_channel = 6;
    int conv1_kernel_size = 5;
    //// C1 feature map
    int C1_channel = conv1_out_channel;
    int C1_size = input_size - (conv1_kernel_size - 1);
    //// S2 feature map
    int S2_channel = C1_channel;
    int S2_size = C1_size / 2;
    //// [Layer] conv2
    int conv2_in_channel = conv1_out_channel;
    int conv2_out_channel = 16;
    int conv2_kernel_size = 5;
    //// C3 feature map
    int C3_channel = conv2_out_channel;
    int C3_size = S2_size - (conv2_kernel_size - 1);
    //// S4 feature map
    int S4_channel = C3_channel;
    int S4_size = C3_size / 2;
    //// [Layer] fc1
    int fc1_in_channel = S4_channel * S4_size * S4_size;
    int fc1_out_channel = 120;
    //// C5 layer
    int C5_size = fc1_out_channel;
    //// [Layer] fc2
    int fc2_in_channel = fc1_out_channel;
    int fc2_out_channel = 84;
    //// F6 layer
    int F6_size = fc2_out_channel;
    //// [Layer] fc3
    int fc3_in_channel = fc2_out_channel;
    int fc3_out_channel = 10;
    //// output
    int output_size = fc3_out_channel;
};

#endif
