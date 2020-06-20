#ifndef LENET5_CPU_H
#define LENET5_CPU_H

#include "LeNet5.h"

class LeNet5_cpu : public LeNet5
{
public:
    // Get from base class
    void load_parameters(std::string value_path) override { LeNet5::load_parameters(value_path); };
    void print_parameters() override { LeNet5::print_parameters(); };
    bool compare(LeNet5* other) override { return LeNet5::compare(other); };
    // Implement!
    LeNet5_cpu(int batch = 1) : LeNet5(batch) {};
    ~LeNet5_cpu() {};
    void predict(const uint8_t* const image, int batch) override;
    void classify(int* predict, int batch) override;
private:
    // Functions
    void normalize(const uint8_t* const image, double* input);
    void relu(double* feature_map, int size);
    void conv(double* input, double* output, double* weight, double* bias,
              int B, int H, int W, int IC, int OC, int K);
    void pool(double* input, double* output,
              int B, int C, int H, int W);
    void fc(double* input, double* output, double* weight, double* bias,
            int B, int IC, int OC);
    // Print Funtions for debug
    void print_fc(double* data, int size);
    void print_C1();
    void print_C3();
};

#endif
