#ifndef COMMON_H
#define COMMON_H

#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>


#define IMG_WIDTH (32)
#define IMG_HEIGHT (32)
#define IMG_CHANNEL (3)
#define IMG_SIZE (IMG_CHANNEL * IMG_WIDTH * IMG_HEIGHT)
#define LABEL_SIZE (1)
#define ROW_SIZE (LABEL_SIZE + IMG_SIZE)

#endif