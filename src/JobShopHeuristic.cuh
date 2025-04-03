#ifndef JOB_SHOP_HEURISTIC_H
#define JOB_SHOP_HEURISTIC_H

#pragma once
#include <cuda_runtime.h>

#ifdef __CUDACC__
#	define GPU_CALLABLE __host__ __device__
#else
#	define GPU_CALLABLE
#endif

#include <string>
#include <vector>

#include "JobShopData.h"
#include "NeuralNetwork.h"

class JobShopHeuristic
{
public:
	// constructor with topology (creating new network)
	JobShopHeuristic(const std::vector<int>& topology);

	// constructor with file (loading network from file)
	JobShopHeuristic(const std::string& filename)
		: neuralNetwork(InitializeNetworkFromFile(filename)) {}

	JobShopHeuristic(NeuralNetwork&& net) : neuralNetwork(std::move(net)) {}

	struct GPUSolution {
		struct OperationSchedule {
			int jobId;		// Job this operation belongs to
			int opId;		// Operation ID
			int startTime;	// Time when the operation starts
			int endTime;	// Time when the operation finishes
		};

		OperationSchedule* operationSchedule;  // Flat array: [machine][operation] instead of vector<vector<OperationSchedule>>
		int* scheduleCounts;
		int makespan = 0;

		// std::vector<std::vector<OperationSchedule>> schedule;  // Every machine's schedule
	};

	Solution Solve(const JobShopData& data);

	NeuralNetwork neuralNetwork;

	void PrintSchedule(const Solution& solution, const JobShopData& data);

	// Add GPU kernel declaration
#ifdef __CUDACC__
	__global__ void SolveProblemsGPU(
		const JobShopData* problems,
		const float* weights,
		GPUSolution* solutions,
		int num_problems,
		int num_weights);
#endif

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