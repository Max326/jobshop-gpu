#ifndef NEURAL_NETWORK_CUH
#define NEURAL_NETWORK_CUH

#pragma once

// i don't know why this is needed, but it is (is it????)
#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __global__
#define __global__
#endif

#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

#include "FileManager.h"

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

	// // GPU-compatible forward pass (weights/biases must be pre-flattened)
	// static __device__ float ForwardGPU(
	// 	const float* weights,  // Flattened [layer][in][out]
	// 	const float* biases,   // Flattened [layer][neuron]
	// 	const int* topology,   // Layer sizes
	// 	const float* input,	   // Input features
	// 	int numLayers);

	void SaveToJson(const std::string& filename) const {
		FileManager::EnsureDataDirExists();
		std::string full_path = FileManager::GetFullPath(filename);

		json j;
		j["topology"] = topology;
		j["weights"] = weights;
		j["biases"] = biases;

		std::ofstream out(full_path);
		if(!out) {
			throw std::runtime_error("Failed to save network to: " + full_path);
		}
		out << j.dump(4);
		std::cout << "Network saved to: " << std::filesystem::absolute(full_path) << std::endl;
	}

	void LoadFromJson(const std::string& filename) {
		std::string full_path = FileManager::GetFullPath(filename);

		if(!std::filesystem::exists(full_path)) {
			throw std::runtime_error("Network file not found: " + full_path);
		}

		std::ifstream in(full_path);
		json j;
		in >> j;
		in.close();

		topology = j["topology"].get<std::vector<int>>();
		weights = j["weights"].get<std::vector<std::vector<float>>>();
		biases = j["biases"].get<std::vector<std::vector<float>>>();

		Validate();			   // Perform validation checks
		InitializeCudaData();  // Initialize CUDA data after loading
	}

	void GenerateWeights();
	void GenerateBiases();

	void FlattenParams();

	struct DeviceEvaluator {
		const float* weights;  // Flattened weights
		const float* biases;
		const int* topology;
		int num_layers;

		__device__ float Evaluate(const float* features) const;
	};

	// Prepares evaluator for GPU
	__host__ DeviceEvaluator GetDeviceEvaluator() const {
		return {
			flattenedWeights.data(),
			flattenedBiases.data(),
			topology.data(),
			(int)topology.size()};
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
	struct CudaData;					 // Forward declaration
	std::unique_ptr<CudaData> cudaData;	 // Cuda data incapsulation

	std::vector<int> topology;
	std::vector<std::vector<float>> weights;
	std::vector<std::vector<float>> biases;
	std::vector<size_t> layerOffsets;
	std::vector<size_t> biasOffsets;

	const float* GetFlattenedWeights() const { return flattenedWeights.data(); }
	const float* GetFlattenedBiases() const { return flattenedBiases.data(); }
	const int* GetTopology() const { return topology.data(); }
	int GetNumLayers() const { return topology.size(); }

private:
	static const int maxLayerSize = 32;
	std::vector<float> flattenedWeights;
	std::vector<float> flattenedBiases;

private:
	void InitializeCudaData();

	void Validate() const {
		if(topology.empty()) {
			throw std::invalid_argument("NeuralNetwork: Topology cannot be empty");
		}

		std::cout << "=== DATA VALIDATION ===\n";
		std::cout << "Topology: ";
		for(auto t: topology)
			std::cout << t << " ";
		std::cout << "\nFirst weight layer: " << weights[0][0] << ", " << weights[0][1] << "...\n";
		std::cout << "First bias: " << biases[0][0] << "\n";
	};
};

#endif	// NEURAL_NETWORK_CUH