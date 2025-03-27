#include "JobShopHeuristic.h"

#include <algorithm>
#include <cfloat>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

JobShopHeuristic::JobShopHeuristic(const std::vector<int>& topology)
	: neuralNetwork(topology) {}  // bezpośrednia inicjalizacja członka

NeuralNetwork JobShopHeuristic::InitializeNetworkFromFile(const std::string& filename) {
	std::string full_path = FileManager::GetFullPath(filename);

	if(!std::filesystem::exists(full_path)) {
		throw std::runtime_error("Network file not found: " + full_path);
	}

	std::ifstream in(full_path);
	if(!in.is_open()) {
		throw std::runtime_error("Cannot open file: " + full_path);
	}

	try {
		json j;
		in >> j;
		in.close();

		std::vector<int> loaded_topology = j["topology"];
		auto weights = j["weights"].get<std::vector<std::vector<float>>>();
		auto biases = j["biases"].get<std::vector<std::vector<float>>>();

		std::cout << "second weight: " << weights[1][2] << std::endl;
		std::cout << "second bias: " << biases[1][0] << std::endl;

		if(loaded_topology.empty() || weights.empty() || biases.empty()) {
			throw std::runtime_error("Invalid network data in file");
		}

		return NeuralNetwork(loaded_topology, &weights, &biases);
	} catch(const std::exception& e) {
		throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
	}
}

JobShopHeuristic::Solution JobShopHeuristic::Solve(const JobShopData& data) {
	Solution solution;
	solution.makespan = 0;
	solution.schedule.resize(data.numMachines);

	// Kopiowanie danych, aby nie modyfikować oryginału
	JobShopData modifiedData = data;

	while(true) {
		// Znajdź dostępne operacje
		bool allScheduled = true;
		float bestScore = -FLT_MAX;
		int bestJobId = -1, bestOperationId = -1, bestMachineId = -1;

		for(int jobId = 0; jobId < modifiedData.numJobs; ++jobId) {
			if(modifiedData.jobs[jobId].operations.empty()) continue;

			int operationId = modifiedData.jobs[jobId].operations.back();
			for(int machineId = 0; machineId < modifiedData.numMachines; ++machineId) {
				if(modifiedData.processingTimes[operationId][machineId] == 0) continue;

				// Przygotuj wektor cech
				std::vector<float> features = ExtractFeatures(modifiedData, jobId, operationId, machineId);

				// std::cout << "Features: " << features[0] << ", " << features[1] << ", " << features[2] << std::endl;

				// Oceń decyzję za pomocą sieci neuronowej
				std::vector<float> output = neuralNetwork.Forward(features);

				// Add to Solve(), after neuralNetwork.Forward()
				// std::cout << "Features: " << features[0] << "," << features[1] << "," << features[2]
				//   << " -> Score: " << output[0] << std::endl;

				float score = output[0];

				if(score > bestScore) {
					bestScore = score;
					bestJobId = jobId;
					bestOperationId = operationId;
					bestMachineId = machineId;
				}
			}
		}

		if(bestJobId == -1) break;	// Wszystkie operacje zaplanowane

		// Zaplanuj operację na maszynie
		UpdateSchedule(modifiedData, bestJobId, bestOperationId, bestMachineId, solution);
	}

	return solution;
}

std::vector<float> JobShopHeuristic::ExtractFeatures(const JobShopData& data, int jobId, int operationId, int machineId) {
	std::vector<float> features;

	// Czas przetwarzania operacji na maszynie
	features.push_back(static_cast<float>(data.processingTimes[operationId][machineId]));

	// Liczba pozostałych operacji w jobie
	features.push_back(static_cast<float>(data.jobs[jobId].operations.size()));

	// Obciążenie maszyny
	int machineLoad = 0;
	for(const auto& op: data.processingTimes) {
		machineLoad += op[machineId];
	}
	features.push_back(static_cast<float>(machineLoad));

	return features;
}

void JobShopHeuristic::UpdateSchedule(JobShopData& data, int jobId, int operationId, int machineId, Solution& solution) {
	// Usuń zaplanowaną operację z joba
	data.jobs[jobId].operations.pop_back();

	// Dodaj operację do harmonogramu maszyny
	solution.schedule[machineId].push_back(operationId);

	// Aktualizuj makespan
	int operationTime = data.processingTimes[operationId][machineId];
	solution.makespan += operationTime;
}