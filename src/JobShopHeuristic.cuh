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

#define MAX_MACHINES 30	 // TODO: make this dynamic
#define MAX_JOBS	 30
#define MAX_OPS		 100

struct OperationSchedule {
	int jobId;
	int opType;
	int startTime;
	int endTime;
};

class SolutionManager
{
public:
	struct GPUSolutions {
		OperationSchedule* allSchedules;  // [machine][operation]
		int* allScheduleCounts;		  // Operations per machine
		int* allMakespans;
		int numProblems;
		int numMachines;
		int maxOps;
	};

	static GPUSolutions CreateGPUSolutions(int numProblems, int numMachines, int maxOpsPerMachine);
	static void FreeGPUSolutions(GPUSolutions& solution);
};

class JobShopHeuristic
{
public:
	struct CPUSolution {
		std::vector<std::vector<OperationSchedule>> schedule;
		int makespan = 0;

		void FromGPU(const SolutionManager::GPUSolutions& gpuSol, int problemId);
		SolutionManager::GPUSolutions ToGPU() const;
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
					SolutionManager::GPUSolutions* solutions,
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
	SolutionManager::GPUSolutions* solutions,
	int total_problems);

#endif	// JOB_SHOP_HEURISTIC_CUH