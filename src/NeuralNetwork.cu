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

	// 2. Validate weights/biases structure
	const size_t num_weight_layers = topology.size() - 1;

	if(weights.size() != num_weight_layers) {
		throw std::invalid_argument("NeuralNetwork: Incorrect number of weight matrices. Expected " +
									std::to_string(num_weight_layers) + ", got " +
									std::to_string(weights.size()));
	}

	if(biases.size() != num_weight_layers) {
		throw std::invalid_argument("NeuralNetwork: Incorrect number of bias vectors. Expected " +
									std::to_string(num_weight_layers) + ", got " +
									std::to_string(biases.size()));
	}

	// 3. Validate individual layer dimensions
	for(size_t i = 0; i < weights.size(); ++i) {
		const int expected_weights = topology[i] * topology[i + 1];
		if(weights[i].size() != static_cast<size_t>(expected_weights)) {
			throw std::invalid_argument("NeuralNetwork: Weights matrix at layer " +
										std::to_string(i) + " has incorrect size. Expected " +
										std::to_string(expected_weights) + ", got " +
										std::to_string(weights[i].size()));
		}
	}

	for(size_t i = 0; i < biases.size(); ++i) {
		const int expected_biases = topology[i + 1];
		if(biases[i].size() != static_cast<size_t>(expected_biases)) {
			throw std::invalid_argument("NeuralNetwork: Bias vector at layer " +
										std::to_string(i) + " has incorrect size. Expected " +
										std::to_string(expected_biases) + ", got " +
										std::to_string(biases[i].size()));
		}
	}

	// 1. Calculate offsets for each layer's weights and biases
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

		size_t sharedMemSize = in_size * sizeof(float); // in_size = current input layer size

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
