#ifndef UTIL_H
#define UTIL_H

#include "common.h"
#include <map>
#include <memory>  // unique_ptr
#include <opencv2/opencv.hpp>

// CIFAR10 Utils
std::map<int, std::string> get_label_dict();
// Image Utils
bool read_image(const std::string data_path, const int index, const int batch,
                uint8_t* const image, int* const label);
bool save_image(const std::string out_path, const uint8_t* const image);

#endif