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
		: neuralNetwork(InitializeNetworkFromFile(filename)) {}

	JobShopHeuristic(NeuralNetwork&& net) : neuralNetwork(std::move(net)) {}

	struct Solution {
		struct OperationSchedule {
			int jobId;		// Job this operation belongs to
			int opId;		// Operation ID
			int startTime;	// Time when the operation starts
			int endTime;	// Time when the operation finishes
		};

		std::vector<std::vector<OperationSchedule>> schedule;  // Every machine's schedule
		int makespan = 0;
	};

	Solution Solve(const JobShopData& data);

	NeuralNetwork neuralNetwork;

	void PrintSchedule(const Solution& solution, const JobShopData& data);

private:
	// JobShopHeuristic(NeuralNetwork&& net) : neuralNetwork(std::move(net)) {}

	static NeuralNetwork InitializeNetworkFromFile(const std::string& filename);

	std::vector<float> ExtractFeatures(const JobShopData& data,
									   const Job& job,
									   const int& operationType,
									   const int& machineId,
									   const int& startTime,
									   const int& machineAvailableTime) const;

	void UpdateSchedule(JobShopData& data, int jobId, int operationId, int machineId, Solution& solution);
};

#endif	// JOB_SHOP_HEURISTIC_H