#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "NeuralNetwork.h"

// Funkcja aktywacji scaleTanh2
__device__ float ScaleTanh2(float x) {
	static const float shift = 3.5f;
	static const float rshift = 1.0f / 3.5f;
	if(x >= 0.f) {
		if(x >= shift)
			return 1.0f + (x - shift) * 0.01;
		float tmp = (x - shift) * rshift;
		return 1.0f - tmp * tmp * tmp * tmp;
	} else if(x >= -shift) {
		float tmp = (x + shift) * rshift;
		return -1.0f + tmp * tmp * tmp * tmp;
	} else {
		return -1.0f - (shift - x) * 0.01;
	}
}

// Kernel CUDA do obliczania wyjścia warstwy
__global__ void ForwardPassKernel(const float *input, const float *weights, const float *biases, float *output, int inputSize, int outputSize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < outputSize) {
		float sum = 0.0f;
		for(int i = 0; i < inputSize; ++i) {
			sum += input[i] * weights[idx * inputSize + i];
		}
		sum += biases[idx];
		output[idx] = ScaleTanh2(sum);	// Zastosowanie funkcji aktywacji
	}
}

NeuralNetwork::NeuralNetwork(const std::vector<int> &topology,
							 const std::vector<std::vector<float>> *weights,
							 const std::vector<std::vector<float>> *biases) : topology(topology) {
	srand(time(0));	 // Inicjalizacja generatora liczb losowych

	// Jeśli wagi zostały podane, użyj ich, w przeciwnym razie zainicjalizuj losowo
	if(weights != nullptr) {
		this->weights = *weights;
	} else {
		for(size_t i = 1; i < topology.size(); ++i) {
			std::vector<float> layerWeights(topology[i] * topology[i - 1]);
			float range = sqrt(6.0f / (topology[i - 1] + topology[i]));	 // Xavier initialization
			for(float &weight: layerWeights) {
				weight = (rand() / (float)RAND_MAX) * 2 * range - range;
			}
			this->weights.push_back(layerWeights);
		}
	}

	// Jeśli biasy zostały podane, użyj ich, w przeciwnym razie zainicjalizuj jako 0
	if(biases != nullptr) {
		this->biases = *biases;
	} else {
		for(size_t i = 1; i < topology.size(); ++i) {
			std::vector<float> layerBiases(topology[i], 0.0f);
			this->biases.push_back(layerBiases);
		}
	}
}

void NeuralNetwork::Forward(const std::vector<float> &input, std::vector<float> &output) {
	// Przeniesienie danych na GPU
	float *d_input, *d_weights, *d_biases, *d_output;
	cudaMalloc(&d_input, input.size() * sizeof(float));
	cudaMalloc(&d_output, topology.back() * sizeof(float));

	cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

	// Obliczenia na GPU dla każdej warstwy
	for(size_t i = 0; i < weights.size(); ++i) {
		cudaMalloc(&d_weights, weights[i].size() * sizeof(float));
		cudaMalloc(&d_biases, biases[i].size() * sizeof(float));

		cudaMemcpy(d_weights, weights[i].data(), weights[i].size() * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_biases, biases[i].data(), biases[i].size() * sizeof(float), cudaMemcpyHostToDevice);

		int blockSize = 256;
		int numBlocks = (topology[i + 1] + blockSize - 1) / blockSize;

		ForwardPassKernel<<<numBlocks, blockSize>>>(d_input, d_weights, d_biases, d_output, topology[i], topology[i + 1]);

		cudaFree(d_weights);
		cudaFree(d_biases);

		// Przeniesienie wyników do następnej warstwy
		if(i < weights.size() - 1) {
			cudaMemcpy(d_input, d_output, topology[i + 1] * sizeof(float), cudaMemcpyDeviceToDevice);
		}
	}

	// Przeniesienie wyników z GPU do CPU
	output.resize(topology.back());
	cudaMemcpy(output.data(), d_output, topology.back() * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
}
