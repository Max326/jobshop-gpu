#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "NeuralNetwork.h"

struct NeuralNetwork::CudaData {
	float *d_weights = nullptr;
	float *d_biases = nullptr;
	float *d_input = nullptr;
	float *d_output = nullptr;
};

// Konstruktor przenoszący
NeuralNetwork::NeuralNetwork(NeuralNetwork &&other) noexcept
	: topology(std::move(other.topology)),
	  weights(std::move(other.weights)),
	  biases(std::move(other.biases)),
	  layerOffsets(std::move(other.layerOffsets)),
	  cudaData(std::move(other.cudaData)) {
	// Zabezpieczenie przed podwójnym zwolnieniem pamięci
	other.cudaData.reset(nullptr);
}

// Operator przypisania przenoszącego
NeuralNetwork &NeuralNetwork::operator=(NeuralNetwork &&other) noexcept {
	if(this != &other) {
		topology = std::move(other.topology);
		weights = std::move(other.weights);
		biases = std::move(other.biases);
		layerOffsets = std::move(other.layerOffsets);
		cudaData = std::move(other.cudaData);
	}
	return *this;
}

// Funkcja aktywacji scaleTanh2
__device__ float ScaleTanh2(float x) {
	constexpr float shift = 3.5f;
	constexpr float rshift = 1.0f / shift;
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

// // Kernel CUDA do obliczania wyjścia warstwy
// __global__ void ForwardPassKernel(const float *input, const float *weights, const float *biases, float *output, int inputSize, int outputSize) {
// 	int idx = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (idx >= outputSize) return;
// 	// DEBUG: Sprawdź pierwszy element
//     if(idx == 0 && threadIdx.x == 0) {
//         printf("GPU: first weight=%f, bias=%f\n", weights[0], biases[0]);
//     }

// 	if(idx < outputSize) {
// 		float sum = 0.0f;
// 		for(int i = 0; i < inputSize; ++i) {
// 			sum += input[i] * weights[idx * inputSize + i];
// 		}
// 		sum += biases[idx];
// 		output[idx] = ScaleTanh2(sum);	// Zastosowanie funkcji aktywacji
// 	}
// }

__global__ void ForwardPassKernel(const float *input, int inputSize, const float *weights, const float *biases, float *output, int outputSize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= outputSize) return;  // Zabezpieczenie!

	float sum = 0.0f;
	if(idx < outputSize) {
		for(int i = 0; i < inputSize; ++i) {
			sum += input[i] * weights[idx * inputSize + i];
		}
		sum += biases[idx];
		output[idx] = ScaleTanh2(sum);	// Zastosowanie funkcji aktywacji
	}
}

NeuralNetwork::NeuralNetwork(const std::vector<int> &topology,
							 const std::vector<std::vector<float>> *weights_ptr,
							 const std::vector<std::vector<float>> *biases_ptr)
	: topology(topology),
	  weights(weights_ptr ? *weights_ptr : std::vector<std::vector<float>>()),
	  biases(biases_ptr ? *biases_ptr : std::vector<std::vector<float>>()),
	  cudaData(std::make_unique<CudaData>()) {
	// 1. Oblicz offsety dla każdej warstwy
	layerOffsets.resize(this->weights.size());
	biasOffsets.resize(biases.size());
	size_t total_weights = 0;
	size_t total_biases = 0;

	for(size_t i = 0; i < this->weights.size(); ++i) {
		layerOffsets[i] = total_weights;
		total_weights += this->weights[i].size();

		biasOffsets[i] = total_biases;
		total_biases += this->biases[i].size();
	}

    // 2. Allocate GPU memory for weights and biases
	CUDA_CHECK(cudaMalloc(&cudaData->d_weights, total_weights * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&cudaData->d_biases, total_biases * sizeof(float)));

	// 3. Find the maximum layer size for input/output buffers
	int max_layer_size = 0;
	for(int size: topology) {
		if(size > max_layer_size) {
			max_layer_size = size;
		}
	}

	// Allocate input and output buffers to the maximum layer size
	CUDA_CHECK(cudaMalloc(&cudaData->d_input, max_layer_size * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&cudaData->d_output, max_layer_size * sizeof(float)));

	// 4. Copy weights and biases to GPU
	size_t weight_offset = 0;
	size_t bias_offset = 0;

	for(size_t i = 0; i < this->weights.size(); ++i) {
		CUDA_CHECK(cudaMemcpy(cudaData->d_weights + weight_offset,
							  this->weights[i].data(),
							  this->weights[i].size() * sizeof(float),
							  cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMemcpy(cudaData->d_biases + bias_offset,
							  this->biases[i].data(),
							  this->biases[i].size() * sizeof(float),
							  cudaMemcpyHostToDevice));

		weight_offset += this->weights[i].size();
		bias_offset += this->biases[i].size();
	}

	// W konstruktorze, po załadowaniu wag:
	std::cout << "=== DATA VALIDATION ===\n";
	std::cout << "Topology: ";
	for(auto t: topology)
		std::cout << t << " ";
	std::cout << "\nFirst weight layer: " << weights[0][0] << ", " << weights[0][1] << "...\n";
	std::cout << "First bias: " << biases[0][0] << "\n";
}

NeuralNetwork::~NeuralNetwork() {
	if(cudaData) {
		cudaFree(cudaData->d_weights);
		cudaFree(cudaData->d_biases);
		cudaFree(cudaData->d_input);
		cudaFree(cudaData->d_output);
	}
}
std::vector<float> NeuralNetwork::Forward(const std::vector<float> &input) {
	// Copy input to device
	CUDA_CHECK(cudaMemcpy(cudaData->d_input, input.data(),
						  input.size() * sizeof(float),
						  cudaMemcpyHostToDevice));

	float *current_input = cudaData->d_input;
	float *current_output = cudaData->d_output;

	for(size_t l = 0; l < weights.size(); ++l) {
		int in_size = topology[l];
		int out_size = topology[l + 1];

		// Get the weight and bias offsets for this layer
		size_t weight_offset = layerOffsets[l];
		size_t bias_offset = biasOffsets[l];

		// Launch the kernel
		int threadsPerBlock = 256;
		int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
		ForwardPassKernel<<<blocksPerGrid, threadsPerBlock>>>(
			current_input, in_size,
			cudaData->d_weights + weight_offset,
			cudaData->d_biases + bias_offset,
			current_output, out_size);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// Swap input and output buffers if not the last layer
		if(l != weights.size() - 1) {
			std::swap(current_input, current_output);
		}
	}

	// Copy the final output from device to host
	std::vector<float> output(topology.back());
	CUDA_CHECK(cudaMemcpy(output.data(), current_output,
						  topology.back() * sizeof(float),
						  cudaMemcpyDeviceToHost));

	return output;
}