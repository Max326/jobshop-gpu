#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "NeuralNetwork.cuh"

__device__ __managed__ int gpu_error_flag = 0;

void NeuralNetwork::InitializeCudaData() {
    // 1. Calculate offsets for each layer's weights and biases
    FlattenParams(); // Upewnij się, że flattenedWeights/Biases są gotowe

    size_t total_weights_bytes = 0;
    if (!flattenedWeights.empty()) { // Użyj flattenedWeights do obliczenia rozmiaru
        total_weights_bytes = flattenedWeights.size() * sizeof(float);
    } else { // Fallback na iterowanie po `weights` jeśli `flattenedWeights` jest puste (np. przed FlattenParams)
        for(const auto& lw : weights) total_weights_bytes += lw.size() * sizeof(float);
    }

    size_t total_biases_bytes = 0;
    if (!flattenedBiases.empty()) { // Użyj flattenedBiases do obliczenia rozmiaru
        total_biases_bytes = flattenedBiases.size() * sizeof(float);
    } else { // Fallback
        for(const auto& lb : biases) total_biases_bytes += lb.size() * sizeof(float);
    }


    if (cudaData->manage_gpu_buffers) {
        // 2. Allocate GPU memory for weights and biases if NN manages them
        if (total_weights_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&cudaData->d_weights, total_weights_bytes));
        }
        if (total_biases_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&cudaData->d_biases, total_biases_bytes));
        }

        // 4. Initialize weight buffers to zero (tylko jeśli alokowane i zarządzane przez NN)
        if (cudaData->d_weights && total_weights_bytes > 0) {
            CUDA_CHECK(cudaMemset(cudaData->d_weights, 0, total_weights_bytes)); // Użyj 0 dla memset, nie 0.0f
        }
        if (cudaData->d_biases && total_biases_bytes > 0) {
            CUDA_CHECK(cudaMemset(cudaData->d_biases, 0, total_biases_bytes)); // Użyj 0 dla memset
        }

        // 5. Copy weights and biases to GPU if NN manages them and host data exists
        if (cudaData->d_weights && !flattenedWeights.empty()) {
            CUDA_CHECK(cudaMemcpy(cudaData->d_weights,
                                  flattenedWeights.data(),
                                  total_weights_bytes,
                                  cudaMemcpyHostToDevice));
        } else if (cudaData->d_weights && !weights.empty() && total_weights_bytes > 0) { // Fallback na kopiowanie warstwa po warstwie
            size_t current_offset = 0;
            for(const auto& lw : weights) {
                if (!lw.empty()) {
                    CUDA_CHECK(cudaMemcpy(cudaData->d_weights + current_offset, lw.data(), lw.size() * sizeof(float), cudaMemcpyHostToDevice));
                    current_offset += lw.size();
                }
            }
        }


        if (cudaData->d_biases && !flattenedBiases.empty()) {
            CUDA_CHECK(cudaMemcpy(cudaData->d_biases,
                                  flattenedBiases.data(),
                                  total_biases_bytes,
                                  cudaMemcpyHostToDevice));
        } else if (cudaData->d_biases && !biases.empty() && total_biases_bytes > 0) { // Fallback
            size_t current_offset = 0;
            for(const auto& lb : biases) {
                if (!lb.empty()) {
                    CUDA_CHECK(cudaMemcpy(cudaData->d_biases + current_offset, lb.data(), lb.size() * sizeof(float), cudaMemcpyHostToDevice));
                    current_offset += lb.size();
                }
            }
        }
    }
    // Jeśli !manage_gpu_buffers, to d_weights i d_biases są ustawiane z zewnątrz.

    // 3. Find the maximum layer size for input/output buffers
    int max_layer_size = 0;
    for(int size: topology) {
        if(size > max_layer_size) {
            max_layer_size = size;
        }
    }
    if (max_layer_size == 0 && !topology.empty()) max_layer_size = topology[0]; // Minimalny fallback
    if (max_layer_size == 0) max_layer_size = 1; // Absolutny minimalny fallback


    // Allocate input and output buffers - te są zawsze zarządzane przez NeuralNetwork
    // Alokuj tylko jeśli jeszcze nie istnieją (np. przy ponownej inicjalizacji)
    if (!cudaData->d_input && max_layer_size > 0) {
        CUDA_CHECK(cudaMalloc(&cudaData->d_input, max_layer_size * sizeof(float)));
    }
    if (!cudaData->d_output && max_layer_size > 0) {
        CUDA_CHECK(cudaMalloc(&cudaData->d_output, max_layer_size * sizeof(float)));
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
        if (cudaData->manage_gpu_buffers) { // Zwalniaj wagi/biasy tylko jeśli NN nimi zarządza
            if(cudaData->d_weights) cudaFree(cudaData->d_weights);
            if(cudaData->d_biases) cudaFree(cudaData->d_biases);
        }
        // d_input i d_output są zawsze zwalniane, bo są zawsze zarządzane przez NN
        if(cudaData->d_input) cudaFree(cudaData->d_input);
        if(cudaData->d_output) cudaFree(cudaData->d_output);
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
	
	if(threadIdx.x == 0 && blockIdx.x == 0) { // don't uncomment, it reduces computing time xD
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
			if(isnan(features[i]) || isinf(features[i])) { //! deleting this slows down the kernel for some fucking reason
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
        
        float next_activations[32]; // Temporary buffer for next layer's activations

        for(int neuron = 0; neuron < out_size; neuron++) {
            float sum = p_shared_biases[bias_idx_offset + neuron]; // Read from shared biases

            for(int i = 0; i < in_size; i++) {
                // Read from shared weights
                float weight_val = p_shared_weights[weight_idx_offset + neuron * in_size + i];
                sum += activations[i] * weight_val;
            }
            next_activations[neuron] = ScaleTanh2(sum); 
        }
        

        for (int i = 0; i < out_size; ++i) {
            if (i < sizeof(activations)/sizeof(float) && i < sizeof(next_activations)/sizeof(float)) { // Sprawdzenie granic odczytu/zapisu
                 activations[i] = next_activations[i];
            }
        }

		//memcpy(activations, next_activations, out_size * sizeof(float));

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