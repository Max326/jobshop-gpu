#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "NeuralNetwork.cuh"

struct NeuralNetwork::CudaData {
	float *d_weights = nullptr;
	float *d_biases = nullptr;
	float *d_input = nullptr;
	float *d_output = nullptr;
};

void NeuralNetwork::InitializeCudaData() {
	// 1. Calculate offsets for each layer's weights and biases
	layerOffsets.resize(weights.size());
	biasOffsets.resize(biases.size());
	size_t total_weights = 0;
	size_t total_biases = 0;

	for(size_t i = 0; i < weights.size(); ++i) {
		layerOffsets[i] = total_weights;
		total_weights += weights[i].size();

		biasOffsets[i] = total_biases;
		total_biases += biases[i].size();
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

	for(size_t i = 0; i < weights.size(); ++i) {
		CUDA_CHECK(cudaMemcpy(cudaData->d_weights + weight_offset,
							  weights[i].data(),
							  weights[i].size() * sizeof(float),
							  cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMemcpy(cudaData->d_biases + bias_offset,
							  biases[i].data(),
							  biases[i].size() * sizeof(float),
							  cudaMemcpyHostToDevice));

		weight_offset += weights[i].size();
		bias_offset += biases[i].size();
	}
}

NeuralNetwork::NeuralNetwork() : cudaData(std::make_unique<CudaData>()) {}

NeuralNetwork::NeuralNetwork(const std::vector<int> &topology,
							 const std::vector<std::vector<float>> *weights_ptr,
							 const std::vector<std::vector<float>> *biases_ptr)
	: topology(topology),
	  weights(weights_ptr ? *weights_ptr : std::vector<std::vector<float>>()),
	  biases(biases_ptr ? *biases_ptr : std::vector<std::vector<float>>()),
	  cudaData(std::make_unique<CudaData>()) {
	// ======================
	//  Validation checks
	// ======================

	if(weights_ptr == nullptr) {
		GenerateWeights();
	}
	if(biases_ptr == nullptr) {
		GenerateBiases();
	}

	// 1. Validate topology
	if(topology.empty()) {
		throw std::invalid_argument("NeuralNetwork: Topology cannot be empty");
	}

	if(topology.size() < 2) {
		throw std::invalid_argument("NeuralNetwork: Topology must have at least 2 layers (input/output)");
	}

	Validate();
	InitializeCudaData();
}

NeuralNetwork::~NeuralNetwork() {
	if(cudaData) {
		cudaFree(cudaData->d_weights);
		cudaFree(cudaData->d_biases);
		cudaFree(cudaData->d_input);
		cudaFree(cudaData->d_output);
	}
}

// Konstruktor przenoszący
NeuralNetwork::NeuralNetwork(NeuralNetwork &&other) noexcept
	: topology(std::move(other.topology)),
	  weights(std::move(other.weights)),
	  biases(std::move(other.biases)),
	  layerOffsets(std::move(other.layerOffsets)),
	  biasOffsets(std::move(other.biasOffsets)),
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
		biasOffsets = std::move(other.biasOffsets);
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

// Example optimized kernel using shared memory
__global__ void ForwardPassKernel(const float *input, int inputSize,
								  const float *weights, const float *biases,
								  float *output, int outputSize) {
	extern __shared__ float shared_input[];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Load input into shared memory (coalesced access)
	if(threadIdx.x < inputSize) {
		shared_input[threadIdx.x] = input[threadIdx.x];
	}
	__syncthreads();

	if(idx < outputSize) {
		float sum = 0.0f;
		for(int i = 0; i < inputSize; ++i) {
			sum += shared_input[i] * weights[idx * inputSize + i];
		}
		sum += biases[idx];
		output[idx] = ScaleTanh2(sum);
	}
}

__device__ float NeuralNetwork::ForwardGPU(
	const float *weights,
	const float *biases,
	const int *topology,
	const float *input,
	int num_layers) {
	float activations[maxLayerSize];  // Adjust based on your topology

	// Copy input
	for(int i = 0; i < topology[0]; i++)
		activations[i] = input[i];

	// Forward pass through layers
	int weight_offset = 0;
	int bias_offset = 0;

	for(int layer = 1; layer < num_layers; layer++) {
		int in_size = topology[layer - 1];
		int out_size = topology[layer];

		for(int neuron = 0; neuron < out_size; neuron++) {
			float sum = biases[bias_offset + neuron];

			for(int input_idx = 0; input_idx < in_size; input_idx++) {
				sum += activations[input_idx] *
					   weights[weight_offset + neuron * in_size + input_idx];
			}

			activations[neuron] = ScaleTanh2(sum);
		}

		weight_offset += in_size * out_size;
		bias_offset += out_size;
	}

	return activations[0];	// Single output
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

		size_t sharedMemSize = in_size * sizeof(float);	 // in_size = current input layer size

		ForwardPassKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
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

void NeuralNetwork::GenerateWeights() {
	weights.clear();
	// if(weights != nullptr) {
	// this->weights = *weights;
	// } else {
	for(size_t i = 1; i < topology.size(); ++i) {
		std::vector<float> layerWeights(topology[i] * topology[i - 1]);
		float range = sqrt(6.0f / (topology[i - 1] + topology[i]));	 // Xavier initialization
		for(float &weight: layerWeights) {
			weight = (rand() / (float)RAND_MAX) * 2 * range - range;
		}
		this->weights.push_back(layerWeights);
	}
	// }
}

void NeuralNetwork::GenerateBiases() {
	biases.clear();
	for(size_t i = 1; i < topology.size(); ++i) {
		std::vector<float> layerBiases(topology[i], 0.1f);
		this->biases.push_back(layerBiases);
	}
}

void NeuralNetwork::FlattenParams() {
	for(const auto& layer: weights) {
		flattenedWeights.insert(flattenedWeights.end(),
								  layer.begin(), layer.end());
	}
	for(const auto& layer: biases) {
		flattenedBiases.insert(flattenedBiases.end(),
								 layer.begin(), layer.end());
	}
}
