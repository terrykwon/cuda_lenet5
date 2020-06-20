#include <stdio.h>
#include <stdlib.h>

__global__ void print_from_gpu(void) {
  printf(
      "Hello World from device!\n\
    threadIdx.x: %d\n\
    blockIdx.x: %d\n\
    blockDim.x: %d\n",
      threadIdx.x, blockIdx.x, blockDim.x);
}

int main(void) {
  printf("Hello World from host!\n");
  print_from_gpu<<<2, 3>>>();
  cudaDeviceSynchronize();
  return 0;
}