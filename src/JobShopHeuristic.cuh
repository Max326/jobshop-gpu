#ifndef JOB_SHOP_HEURISTIC_CUH
#define JOB_SHOP_HEURISTIC_CUH

#pragma once

#ifdef __CUDACC__
#	define GPU_CALLABLE __host__ __device__
#	include <cuda_runtime.h>
#else
#	define GPU_CALLABLE
#endif

#include <cfloat>
#include <memory>
#include <string>
#include <vector>

#include "JobShopData.cuh"
#include "NeuralNetwork.cuh"

#define MAX_MACHINES 30
#define MAX_JOBS	 30
#define MAX_OPS		 1000

struct OperationSchedule {
	int jobId;
	int opId;
	int startTime;
	int endTime;
};

class SolutionManager
{
public:
	struct GPUSolution {
		OperationSchedule* schedule;  // [machine][operation]
		int* scheduleCounts;		  // Operations per machine
		int makespan;
		int numMachines;
	};

	static GPUSolution CreateGPUSolution(int numMachines, int maxOpsPerMachine);
	static void FreeGPUSolution(GPUSolution& solution);
};

// struct GPUSolverState {
// 	// Problem state
// 	int* machine_available_times;
// 	int* job_next_op;
// 	int* job_last_end;

// 	// Solution tracking
// 	OperationSchedule* schedule;
// 	int* schedule_counts;
// 	int makespan;
// };

class JobShopHeuristic
{
public:
	struct CPUSolution {
		std::vector<std::vector<OperationSchedule>> schedule;
		int makespan = 0;

		void FromGPU(const SolutionManager::GPUSolution& gpuSol);
		SolutionManager::GPUSolution ToGPU() const;
	};

	CPUSolution Solve(const JobShopData& data);

	// constructor with topology (creating new network)
	JobShopHeuristic(const std::vector<int>& topology);

	// constructor with file (loading network from file)
	JobShopHeuristic(const std::string& filename);

	JobShopHeuristic(NeuralNetwork&& net);

	// GPU Interface
	struct SolverConfig {
		int maxThreadsPerBlock = 256;
		int maxOperationsPerMachine = 100;
	};

	void SolveBatch(const GPUProblem* problems,
					SolutionManager::GPUSolution* solutions,
					int numProblems);

	void PrintSchedule(const CPUSolution& solution, const JobShopData& data);

public:
	NeuralNetwork neuralNetwork;

private:
	static NeuralNetwork InitializeNetworkFromFile(const std::string& filename);

	std::vector<float> ExtractFeatures(const JobShopData& data,
									   const Job& job,
									   const int& operationType,
									   const int& machineId,
									   const int& startTime,
									   const int& machineAvailableTime) const;

	void UpdateSchedule(JobShopData& data, int jobId, int operationId, int machineId, CPUSolution& solution);
};

__global__ void SolveFJSSPKernel(
	const GPUProblem* problems,
	const NeuralNetwork::DeviceEvaluator nn_eval,
	SolutionManager::GPUSolution* solutions,
	int total_problems);

#endif	// JOB_SHOP_HEURISTIC_CUH