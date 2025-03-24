#ifndef JOB_SHOP_HEURISTIC_H
#define JOB_SHOP_HEURISTIC_H

#pragma once

#include <string>
#include <vector>

#include "JobShopData.h"
#include "NeuralNetwork.h"

class JobShopHeuristic
{
public:
	// Konstruktor z topologią (tworzy nową sieć)
	JobShopHeuristic(const std::vector<int>& topology);

	// Konstruktor ładujący z pliku
	JobShopHeuristic(const std::string& filename)
		: JobShopHeuristic(InitializeNetworkFromFile(filename)) {}

	struct Solution {
		double makespan;
		std::vector<std::vector<int>> schedule;	 // Harmonogram dla każdej maszyny
	};

	Solution Solve(const JobShopData& data);

	NeuralNetwork neuralNetwork;

private:
	JobShopHeuristic(NeuralNetwork&& net) : neuralNetwork(std::move(net)) {}

	static NeuralNetwork InitializeNetworkFromFile(const std::string& filename);

	std::vector<float> ExtractFeatures(const JobShopData& data, int jobId, int operationId, int machineId);
	void UpdateSchedule(JobShopData& data, int jobId, int operationId, int machineId, Solution& solution);
};

#endif	// JOB_SHOP_HEURISTIC_H