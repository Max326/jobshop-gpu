#include <stdio.h>

// Funkcja kernel
__global__ void addVectors(float* A, float* B, float* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) { // Przykładowa granica
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Dane wejściowe
    const int N = 1024;
    float* host_A, *host_B, *host_C;
    float* device_A, *device_B, *device_C;

    // Alokacja pamięci na CPU
    host_A = (float*)malloc(N * sizeof(float));
    host_B = (float*)malloc(N * sizeof(float));
    host_C = (float*)malloc(N * sizeof(float));

    // Inicjacja danych
    for (int i = 0; i < N; i++) {
        host_A[i] = i;
        host_B[i] = i * 2;
    }

    // Alokacja pamięci na GPU
    cudaMalloc((void**)&device_A, N * sizeof(float));
    cudaMalloc((void**)&device_B, N * sizeof(float));
    cudaMalloc((void**)&device_C, N * sizeof(float));

    // Przeniesienie danych na GPU
    cudaMemcpy(device_A, host_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Wywołanie kernela
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    addVectors<<<numBlocks, blockSize>>>(device_A, device_B, device_C);

    // Przeniesienie wyników z powrotem na CPU
    cudaMemcpy(host_C, device_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Wypisanie wyników
    for (int i = 0; i < N; i++) {
        printf("%f + %f = %f\n", host_A[i], host_B[i], host_C[i]);
    }

    // Zwolnienie pamięci
    free(host_A);
    free(host_B);
    free(host_C);
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    return 0;
}
