#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "NeuralNetwork.cuh"


void NeuralNetwork::InitializeCudaData() {
	// 1. Calculate offsets for each layer's weights and biases
	FlattenParams();

	std::cout << "Flattened weights size: " << flattenedWeights.size() << "\n";
	std::cout << "Flattened biases size: " << flattenedBiases.size() << "\n";
	std::cout << "First few weights: ";
	for(int i = 0; i < std::min(5, (int)flattenedWeights.size()); i++)
		std::cout << flattenedWeights[i] << " ";
	std::cout << "\n";

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

__device__ float NeuralNetwork::DeviceEvaluator::Evaluate(const float *features) const {
	const int MAX_LAYER_SIZE = 32;	// Match your header definition
	float activations[MAX_LAYER_SIZE];

	// 1. Validate input size
	if(topology[0] > MAX_LAYER_SIZE) return 0.0f;

	// 2. Copy input with bounds checking
	for(int i = 0; i < topology[0] && i < MAX_LAYER_SIZE; i++) {
		activations[i] = features[i];
	}

	int weight_offset = 0;
	int bias_offset = 0;
	int total_weights = 0;
	int total_biases = 0;

	// 3. Pre-calculate total weights/biases for bounds checking
	for(int i = 1; i < num_layers; i++) {
		total_weights += topology[i - 1] * topology[i];
		total_biases += topology[i];
	}

	for(int layer = 1; layer < num_layers; layer++) {
		int in_size = topology[layer - 1];
		int out_size = topology[layer];

		// 4. Validate layer dimensions
		if(out_size > MAX_LAYER_SIZE) return 0.0f;

		for(int neuron = 0; neuron < out_size; neuron++) {
			// 5. Check bias access
			if(bias_offset + neuron >= total_biases) {
				printf("Bias access out of bounds: %d >= %d\n",
					   bias_offset + neuron, total_biases);
				return 0.0f;
			}

			float sum = biases[bias_offset + neuron];

			for(int i = 0; i < in_size; i++) {
				// 6. Check weight access
				int weight_idx = weight_offset + neuron * in_size + i;
				if(weight_idx >= total_weights) {
					printf("Weight access out of bounds: %d >= %d\n",
						   weight_idx, total_weights);
					return 0.0f;
				}

				sum += activations[i] * weights[weight_idx];
			}

			activations[neuron] = ScaleTanh2(sum);
		}

		weight_offset += in_size * out_size;
		bias_offset += out_size;
	}

	return activations[0];
}

void NeuralNetwork::GenerateWeights() {
	weights.clear();
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
	flattenedWeights.clear();
	flattenedBiases.clear();

	for(const auto &layer: weights) {
		flattenedWeights.insert(flattenedWeights.end(),
								layer.begin(), layer.end());
	}

	for(const auto &layer: biases) {
		flattenedBiases.insert(flattenedBiases.end(),
							   layer.begin(), layer.end());
	}
}