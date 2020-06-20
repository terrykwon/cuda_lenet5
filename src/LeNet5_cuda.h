#ifndef LENET5_CUDA_H
#define LENET5_CUDA_H

#include "LeNet5.h"

class LeNet5_cuda : public LeNet5
{
public:
    // Get from base class
    void load_parameters(std::string value_path) override { LeNet5::load_parameters(value_path); };
    void print_parameters() override { LeNet5::print_parameters(); };
    bool compare(LeNet5* other) override { return LeNet5::compare(other); };
    void prepare_device_memory(uint8_t* image); 
    // Implement!
    LeNet5_cuda(int batch = 1) : LeNet5(batch) {};
    void predict(int batch) ;
    void predict(const uint8_t* const image, int batch) override {predict(batch);}
    void classify(int* predict, int batch) override;
    ~LeNet5_cuda();
private:
    //////////////////////////////////////////////////
    //Device Weights 
    //////////////////////////////////////////////////
    double* d_conv1_weight;   // [3][6][5][5];
    double* d_conv2_weight;   // [6][16][5][5];
    double* d_conv1_bias;     // [6];
    double* d_conv2_bias;     // [16];
    double* d_fc1_weight;     // [400][120];
    double* d_fc2_weight;     // [120][84];
    double* d_fc3_weight;     // [84][10];
    double* d_fc1_bias;       // [120];
    double* d_fc2_bias;       // [84];
    double* d_fc3_bias;       // [10];
    //////////////////////////////////////////////////
    // Device Feature Maps
    //////////////////////////////////////////////////
    uint8_t* d_image;          // [batch][3][32][32];
    double* d_input;          // [batch][3][32][32];
    double* d_C1_feature_map; // [batch][6][28][28];
    double* d_S2_feature_map; // [batch][6][14][14];
    double* d_C3_feature_map; // [batch][16][10][10];
    double* d_S4_feature_map; // [batch][16][5][5];
    double* d_C5_layer;       // [batch][120];
    double* d_F6_layer;       // [batch][84];
    double* d_output;         // [batch][10];
    int*    d_predict_cuda;   // [batch];

    // Functions
    // __global__ void normalize(int batch, int input_channel, int input_size, const uint8_t* const d_image, double* d_input);

     // CPU Functions
    void cpu_normalize(const uint8_t* const image, double* input);
    void cpu_relu(double* feature_map, int size);
    void cpu_conv(double* input, double* output, double* weight, double* bias,
              int B, int H, int W, int IC, int OC, int K);
    void cpu_pool(double* input, double* output,
         int B, int C, int H, int W);
    void cpu_fc(double* input, double* output, double* weight, double* bias,
         int B, int IC, int OC);
    void cpu_softmax(double* input, int* output, int B, int size);
    // P cpu_int Funtions for debug
    void cpu_print_output(double* data) {
      for(int b = 0;b<batch;b++) {
        for (int i=0;i<output_size;i++) {
        printf("[%d][%d]: %lf\n", b,i,data[b*output_size + i]);
        }
      }
    }

 
};

#endif
