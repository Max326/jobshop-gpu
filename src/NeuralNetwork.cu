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
							 const std::vector<std::vector<float>> *weights_ptr,
							 const std::vector<std::vector<float>> *biases_ptr)
	: topology(topology),
	  weights(weights_ptr ? *weights_ptr : std::vector<std::vector<float>>()),
	  biases(biases_ptr ? *biases_ptr : std::vector<std::vector<float>>()),
	  cudaData(std::make_unique<CudaData>()) {
	// 1. Oblicz offsety dla każdej warstwy
	layerOffsets.resize(this->weights.size());
	size_t total_weights = 0;
	size_t total_biases = 0;

	for(size_t i = 0; i < this->weights.size(); ++i) {
		layerOffsets[i] = total_weights;
		total_weights += this->weights[i].size();
		total_biases += this->biases[i].size();
	}

	// 2. Alokacja pamięci GPU
	cudaMalloc(&cudaData->d_weights, total_weights * sizeof(float));
	cudaMalloc(&cudaData->d_biases, total_biases * sizeof(float));
	cudaMalloc(&cudaData->d_input, topology[0] * sizeof(float));
	cudaMalloc(&cudaData->d_output, topology.back() * sizeof(float));

	// 3. Skopiuj wagi i biasy
	size_t weight_offset = 0;
	size_t bias_offset = 0;

	for(size_t i = 0; i < this->weights.size(); ++i) {
		cudaMemcpy(cudaData->d_weights + weight_offset,
				   this->weights[i].data(),
				   this->weights[i].size() * sizeof(float),
				   cudaMemcpyHostToDevice);

		cudaMemcpy(cudaData->d_biases + bias_offset,
				   this->biases[i].data(),
				   this->biases[i].size() * sizeof(float),
				   cudaMemcpyHostToDevice);

		weight_offset += this->weights[i].size();
		bias_offset += this->biases[i].size();
	}
}

NeuralNetwork::~NeuralNetwork() {
	if(cudaData) {
		cudaFree(cudaData->d_weights);
		cudaFree(cudaData->d_biases);
		cudaFree(cudaData->d_input);
		cudaFree(cudaData->d_output);
	}
}
void NeuralNetwork::Forward(const std::vector<float> &input, std::vector<float> &output) {
	// 1. Kopiuj wejście
	cudaMemcpy(cudaData->d_input, input.data(),
			   input.size() * sizeof(float),
			   cudaMemcpyHostToDevice);

	// 2. Obliczenia dla każdej warstwy
	size_t weight_offset = 0;
	size_t bias_offset = 0;

	for(size_t i = 0; i < weights.size(); ++i) {
		dim3 block(256);
		dim3 grid((topology[i + 1] + block.x - 1) / block.x);

		ForwardPassKernel<<<grid, block>>>(
			i == 0 ? cudaData->d_input : cudaData->d_output,
			cudaData->d_weights + weight_offset,
			cudaData->d_biases + bias_offset,
			cudaData->d_output,
			topology[i],
			topology[i + 1]);

		weight_offset += weights[i].size();
		bias_offset += biases[i].size();
	}

	// 3. Pobierz wynik
	output.resize(topology.back());
	cudaMemcpy(output.data(), cudaData->d_output,
			   output.size() * sizeof(float),
			   cudaMemcpyDeviceToHost);
}