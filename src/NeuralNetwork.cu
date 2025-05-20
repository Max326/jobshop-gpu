#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "NeuralNetwork.cuh"

__device__ __managed__ int gpu_error_flag = 0;

void NeuralNetwork::InitializeCudaData() {
	// 1. Calculate offsets for each layer's weights and biases
	FlattenParams();

	/* std::cout << "\nFlattened weights size: " << flattenedWeights.size() << "\n";
	std::cout << "Flattened biases size: " << flattenedBiases.size() << "\n"; */

	/* std::cout << "First few weights: ";
	for(int i = 0; i < std::min(5, (int)flattenedWeights.size()); i++)
		std::cout << flattenedWeights[i] << " ";
		std::cout << "\n"; */

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

	// 4. Initialize weight buffers to zero
	CUDA_CHECK(cudaMemset(cudaData->d_weights, 0.0f, total_weights * sizeof(float)));
	CUDA_CHECK(cudaMemset(cudaData->d_biases, 0.0f, total_biases * sizeof(float)));

	// 5. Copy weights and biases to GPU
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
	// Sprawdź, czy wejście jest NaN lub Inf
	if(isnan(x) || isinf(x)) {
		printf("[ERROR] ScaleTanh2 received invalid input: %f\n", x);
		return 0.0f;
	}

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
	const int MAX_LAYER_SIZE = maxLayerSize;
	float activations[MAX_LAYER_SIZE];

	// if (this->max_layer_size <= 0 || this->max_layer_size > 101 /*Match static const*/) { // Basic sanity check
    //     return 0.0f; // Or handle error differently if critical path allows
    // }
	if(threadIdx.x == 0 && blockIdx.x == 0) {
		for(int i = 0; i < this->d_topology[0]; i++) {
			if(isnan(features[i]) || isinf(features[i])) {
				printf("[ERROR] Invalid input feature at index %d: %f\n", i, features[i]);
				NeuralNetwork::DeviceEvaluator::ReportAndAbort("Invalid input feature");
				return 0.0f;
			}
		}
	}
	
	// 2. Copy input (without printing)
	for(int i = 0; i < this->d_topology[0]; i++) {
		activations[i] = features[i];
	}

	int weight_offset = 0;
	int bias_offset = 0;

	// Calculate totals without printing
	int total_weights_for_eval = 0;
	int total_biases_for_eval = 0;
	for(int i = 1; i < this->num_layers; i++) {
		total_weights_for_eval += this->d_topology[i - 1] * this->d_topology[i];
		total_biases_for_eval += this->d_topology[i];
	}

	for(int layer = 1; layer < this->num_layers; layer++) {
		int in_size = this->d_topology[layer - 1];
		int out_size = this->d_topology[layer];


		for(int neuron = 0; neuron < out_size; neuron++) {
			float sum = this->biases[bias_offset + neuron];

			for(int i = 0; i < in_size; i++) {
				int weight_idx = weight_offset + neuron * in_size + i;

				sum += activations[i] * this->weights[weight_idx];
			}

			activations[neuron] = ScaleTanh2(sum);
		}

		weight_offset += in_size * out_size;
		bias_offset += out_size;
	}

	float final_output = (this->d_topology[this->num_layers - 1] == 1) ? activations[0] : 0.0f;

	return final_output;
}

// New Evaluate function using shared memory pointers
__device__ float NeuralNetwork::DeviceEvaluator::Evaluate(const float* features, const float* p_shared_weights, const float* p_shared_biases) const {
    // Use this->max_layer_size which is now set correctly

    if (this->max_layer_size <= 0 || this->max_layer_size > 101 /*Match static const*/) { // Basic sanity check
        return 0.0f; // Or handle error differently if critical path allows
    }
    // Using 101 directly as per existing code's use of NeuralNetwork::maxLayerSize
    float activations[maxLayerSize]; // Max size for activations array on stack

	if(threadIdx.x == 0 && blockIdx.x == 0) {
		for(int i = 0; i < this->d_topology[0]; i++) {
			if(isnan(features[i]) || isinf(features[i])) {
				printf("[ERROR] Invalid input feature at index %d: %f\n", i, features[i]);
				NeuralNetwork::DeviceEvaluator::ReportAndAbort("Invalid input feature");
				return 0.0f;
			}
		}
	}

    // Input features copy (checks removed for performance as per your successful test)
    for(int i = 0; i < this->d_topology[0]; i++) {
        activations[i] = features[i];
    }

    int weight_idx_offset = 0; // Offset for reading from p_shared_weights
    int bias_idx_offset = 0;   // Offset for reading from p_shared_biases

    for(int layer = 1; layer < this->num_layers; layer++) {
        int in_size = this->d_topology[layer - 1];
        int out_size = this->d_topology[layer];
        
        float next_activations[101]; // Temporary buffer for next layer's activations

        for(int neuron = 0; neuron < out_size; neuron++) {
            float sum = p_shared_biases[bias_idx_offset + neuron]; // Read from shared biases

            for(int i = 0; i < in_size; i++) {
                // Read from shared weights
                float weight_val = p_shared_weights[weight_idx_offset + neuron * in_size + i];
                sum += activations[i] * weight_val;
            }
            next_activations[neuron] = ScaleTanh2(sum); 
        }
        

        // for(int i=0; i < out_size; ++i) { // Copy to activations for next layer
        //     activations[i] = next_activations[i];
        // }

		memcpy(activations, next_activations, out_size * sizeof(float));

        weight_idx_offset += in_size * out_size;
        bias_idx_offset += out_size;
    }

    // Assuming output layer has 1 neuron for FJSS evaluation score
    float final_output = (this->d_topology[this->num_layers - 1] == 1) ? activations[0] : 0.0f;
    return final_output;
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

std::vector<NeuralNetwork> NeuralNetwork::LoadBatchFromJson(const std::string &filename) {
	std::string full_path = FileManager::GetFullPath(filename);
	std::ifstream in(full_path);
	if(!in) throw std::runtime_error("Cannot open weights file: " + full_path);

	nlohmann::json all_nets;
	in >> all_nets;
	in.close();

	std::vector<NeuralNetwork> networks;
	for(const auto &j: all_nets) {
		std::vector<int> topology = j["topology"].get<std::vector<int>>();
		std::vector<std::vector<float>> weights = j["weights"].get<std::vector<std::vector<float>>>();
		std::vector<std::vector<float>> biases = j["biases"].get<std::vector<std::vector<float>>>();
		networks.emplace_back(topology, &weights, &biases);
	}
	return networks;
}