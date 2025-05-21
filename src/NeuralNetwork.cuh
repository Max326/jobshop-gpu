#ifndef NEURAL_NETWORK_CUH
#define NEURAL_NETWORK_CUH

#pragma once

#ifndef __host__
#	define __host__
#endif

#ifndef __device__
#	define __device__
#endif

#ifndef __global__
#	define __global__
#endif

#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

#include "FileManager.h"

#define MAX_NN_LAYERS 4

extern __device__ __managed__ int gpu_error_flag;

class NeuralNetwork
{
public:
	using json = nlohmann::json;

	NeuralNetwork();

	NeuralNetwork(const std::vector<int>& topology,
				  const std::vector<std::vector<float>>* weights = nullptr,
				  const std::vector<std::vector<float>>* biases = nullptr);

	NeuralNetwork(NeuralNetwork&& other) noexcept;
	NeuralNetwork& operator=(NeuralNetwork&& other) noexcept;

	NeuralNetwork(const NeuralNetwork&) = default;

	~NeuralNetwork();

	std::vector<float> Forward(const std::vector<float>& input);

	void GenerateWeights();
	void GenerateBiases();

	void FlattenParams();

	static int CalculateTotalParameters(const std::vector<int>& topology) {
		int total = 0;
		for(size_t i = 1; i < topology.size(); ++i) {
			// Wagi i biasy dla warstwy
			total += topology[i - 1] * topology[i] + topology[i];
		}
		return total;
	}

	
	struct DeviceEvaluator {
		const float* weights;			// Flattened weights on device
		const float* biases;			// Flattened biases on device
		int d_topology[MAX_NN_LAYERS];	// Embedded topology array
		int num_layers;
		int max_layer_size;

		__device__ void ReportAndAbort(const char* msg) const {
			atomicExch(&gpu_error_flag, 1);
			__threadfence_system();
			asm("trap;");
		}

		__device__ float Evaluate(const float* features, const float* p_shared_weights, const float* p_shared_biases) const;
	};

	__host__ DeviceEvaluator GetDeviceEvaluator() const {
		if(!cudaData || !cudaData->d_weights || !cudaData->d_biases) {
			throw std::runtime_error("CUDA data not initialized for GetDeviceEvaluator");
		}
		if(topology.size() > MAX_NN_LAYERS) {
			throw std::runtime_error("Network topology exceeds MAX_NN_LAYERS defined in DeviceEvaluator.");
		}

		DeviceEvaluator eval;
		eval.weights = cudaData->d_weights;
		eval.biases = cudaData->d_biases;

		for(size_t i = 0; i < topology.size(); ++i) {
			eval.d_topology[i] = topology[i];
		}
		// Pad remaining elements if topology.size() < MAX_NN_LAYERS
		for(size_t i = topology.size(); i < MAX_NN_LAYERS; ++i) {
			eval.d_topology[i] = 0;
		}
		eval.num_layers = static_cast<int>(topology.size());
		eval.max_layer_size = NeuralNetwork::maxLayerSize;

		return eval;
	}
#define CUDA_CHECK(call)                                                                                \
	{                                                                                                   \
		cudaError_t err = (call);                                                                       \
		if(err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
			exit(1);                                                                                    \
		}                                                                                               \
	}

public:
	struct CudaData {
		float* d_weights = nullptr;
		float* d_biases = nullptr;
		float* d_input = nullptr;
		float* d_output = nullptr;
	};	// Forward declaration
	std::unique_ptr<CudaData> cudaData;	 // Cuda data incapsulation

	std::vector<int> topology;
	std::vector<std::vector<float>> weights;
	std::vector<std::vector<float>> biases;

	const int* GetTopology() const { return topology.data(); }
	int GetNumLayers() const { return topology.size(); }
	static const int maxLayerSize = 86;


private:
	void InitializeCudaData();

	void Validate(bool talk = false) const {
		if(topology.empty()) {
			throw std::invalid_argument("NeuralNetwork: Topology cannot be empty");
		}

		if(talk) {
			std::cout << "=== DATA VALIDATION ===\n";
			std::cout << "Topology: ";
			for(auto t: topology)
				std::cout << t << " ";
		}
	};
};

#endif	// NEURAL_NETWORK_CUH