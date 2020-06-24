#ifndef LENET5_CUDA_H
#define LENET5_CUDA_H

#include "LeNet5.h"

// Conv params in constant memory
__constant__ float d_conv1_weight[3*6*5*5];   // [3][6][5][5];
__constant__ float d_conv2_weight[6*16*5*5];   // [6][16][5][5];
__constant__ float d_conv1_bias[6];     // [6];
__constant__ float d_conv2_bias[16];     // [16];

class LeNet5_cuda : public LeNet5
{
public:
    // Get from base class
    void load_parameters(std::string value_path) override { LeNet5::load_parameters(value_path); };
    void print_parameters() override { LeNet5::print_parameters(); };
    bool compare(LeNet5* other) override { return LeNet5::compare(other); };
    void prepare_device_memory(uint8_t* image); 
    // Implement!
    // LeNet5_cuda(int batch = 1) : LeNet5(batch) {};
    LeNet5_cuda(int batch = 1);
    void predict(int batch) ;
    void predict(const uint8_t* const image, int batch) override {predict(batch);}
    void classify(int* predict, int batch) override;
    ~LeNet5_cuda();
private:
    //////////////////////////////////////////////////
    //Device Weights 
    //////////////////////////////////////////////////
    float* d_fc1_weight;     // [400][120];
    float* d_fc2_weight;     // [120][84];
    float* d_fc3_weight;     // [84][10];
    float* d_fc1_bias;       // [120];
    float* d_fc2_bias;       // [84];
    float* d_fc3_bias;       // [10];
    //////////////////////////////////////////////////
    // Device Feature Maps
    //////////////////////////////////////////////////
    uint8_t* d_image;          // [batch][3][32][32];
    float* d_input;          // [batch][3][32][32];
    float* d_C1_feature_map; // [batch][6][28][28];
    float* d_S2_feature_map; // [batch][6][14][14];
    float* d_C3_feature_map; // [batch][16][10][10];
    float* d_S4_feature_map; // [batch][16][5][5];
    float* d_C5_layer;       // [batch][120];
    float* d_F6_layer;       // [batch][84];
    float* d_output;         // [batch][10];
    int*    d_predict_cuda;   // [batch];

    // Float host params 
    float* f_conv1_weight;   // [3][6][5][5];
    float* f_conv2_weight;   // [6][16][5][5];
    float* f_conv1_bias;     // [6];
    float* f_conv2_bias;     // [16];
    float* f_fc1_weight;     // [400][120];
    float* f_fc2_weight;     // [120][84];
    float* f_fc3_weight;     // [84][10];
    float* f_fc1_bias;       // [120];
    float* f_fc2_bias;       // [84];
    float* f_fc3_bias;       // [10];

    // // Float host feature maps
    // float* f_input;          // [batch][3][32][32];
    // float* f_C1_feature_map; // [batch][6][28][28];
    // float* f_S2_feature_map; // [batch][6][14][14];
    // float* f_C3_feature_map; // [batch][16][10][10];
    // float* f_S4_feature_map; // [batch][16][5][5];
    // float* f_C5_layer;       // [batch][120];
    // float* f_F6_layer;       // [batch][84];
    float* f_output;         // [batch][10];

    // For unrolled convolution
    float* d_input_unrolled;
    float* d_conv1_weight_unrolled;
    float* d_conv1_bias_unrolled; // is this necessary?
    float* d_S2_feature_map_unrolled;
    float* d_conv2_weight_unrolled;
    float* d_conv2_bias_unrolled;

    // Functions
    // __global__ void normalize(int batch, int input_channel, int input_size, const uint8_t* const d_image, float* d_input);

     // CPU Functions
    void cpu_normalize(const uint8_t* const image, float* input);
    void cpu_relu(float* feature_map, int size);
    void cpu_conv(float* input, float* output, float* weight, float* bias,
              int B, int H, int W, int IC, int OC, int K);
    void cpu_pool(float* input, float* output,
         int B, int C, int H, int W);
    void cpu_fc(float* input, float* output, float* weight, float* bias,
         int B, int IC, int OC);
    void cpu_softmax(float* input, int* output, int B, int size);
    // P cpu_int Funtions for debug
    void cpu_print_output(float* data) {
      for(int b = 0;b<batch;b++) {
        for (int i=0;i<output_size;i++) {
        printf("[%d][%d]: %lf\n", b,i,data[b*output_size + i]);
        }
      }
    }

 
};

#endif
