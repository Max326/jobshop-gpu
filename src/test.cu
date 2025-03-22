#include <iostream>
#include <cuda_runtime.h>

__global__ void helloCUDA() {
    printf("Hello from GPU!\n");
}

int main() {
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
    std::cout << "Hello from CPU!" << std::endl;
    return 0;
}