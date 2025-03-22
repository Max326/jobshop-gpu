#include <iostream>
#include <cuda_runtime.h>

__global__ void dotProduct(float *x, float *w, float *result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        result[tid] = x[tid] * w[tid];
    }
}

int main() {
    const int N = 1024;
    float h_x[N], h_w[N], h_result[N];  // Dane na CPU
    float *d_x, *d_w, *d_result;        // Dane na GPU

    // Inicjalizacja danych na CPU
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;  // Przykładowe dane wejściowe
        h_w[i] = 0.5f;  // Przykładowe wagi
    }

    // Alokacja pamięci na GPU
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_w, N * sizeof(float));
    cudaMalloc((void**)&d_result, N * sizeof(float));

    // Kopiowanie danych do GPU
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, N * sizeof(float), cudaMemcpyHostToDevice);

    // Uruchomienie kernela
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    dotProduct<<<gridSize, blockSize>>>(d_x, d_w, d_result, N);

    // Kopiowanie wyników z GPU do CPU
    cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Sumowanie wyników (na CPU)
    float y = 0.0f;
    for (int i = 0; i < N; i++) {
        y += h_result[i];
    }
    y += 1.0f;  // Dodanie biasu

    // Wyświetlenie wyniku
    std::cout << "Output: " << y << std::endl;

    // Zwolnienie pamięci na GPU
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_result);

    return 0;
}