#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#pragma once
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

#include "FileManager.h"

class NeuralNetwork
{
public:
	using json = nlohmann::json;

	NeuralNetwork(const std::vector<int>& topology,
				  const std::vector<std::vector<float>>* weights = nullptr,
				  const std::vector<std::vector<float>>* biases = nullptr);

	NeuralNetwork(NeuralNetwork&& other) noexcept;
	NeuralNetwork& operator=(NeuralNetwork&& other) noexcept;

	NeuralNetwork(const NeuralNetwork&) = delete;
	NeuralNetwork& operator=(const NeuralNetwork&) = delete;

	~NeuralNetwork();

	void Forward(const std::vector<float>& input, std::vector<float>& output);

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
	}

private:
	struct CudaData;					 // Forward declaration
	std::unique_ptr<CudaData> cudaData;	 // Enkapsulacja danych CUDA

	std::vector<int> topology;
	std::vector<std::vector<float>> weights;
	std::vector<std::vector<float>> biases;
	std::vector<size_t> layerOffsets;
};

#endif	// NEURAL_NETWORK_H