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

struct GPUSolverState {
	// Problem state
	int* machine_available_times;
	int* job_next_op;
	int* job_last_end;

	// Solution tracking
	OperationSchedule* schedule;
	int* schedule_counts;
	int makespan;
};

__global__ void SolveFJSSPKernel(
	const GPUProblem* problems,
	const NeuralNetwork::DeviceEvaluator nn_eval,
	SolutionManager::GPUSolution* solutions,
	int total_problems);

class JobShopHeuristic
{
public:
	// constructor with topology (creating new network)
	JobShopHeuristic(const std::vector<int>& topology);

	// constructor with file (loading network from file)
	JobShopHeuristic(const std::string& filename)
		: neuralNetwork(InitializeNetworkFromFile(filename)) {}

	JobShopHeuristic(NeuralNetwork&& net) : neuralNetwork(std::move(net)) {}

	struct CPUSolution {
		std::vector<std::vector<OperationSchedule>> schedule;
		int makespan = 0;

		void FromGPU(const SolutionManager::GPUSolution& gpuSol);
		SolutionManager::GPUSolution ToGPU() const;
	};

	CPUSolution Solve(const JobShopData& data);

	// GPU Interface
	struct SolverConfig {
		int maxThreadsPerBlock = 256;
		int maxOperationsPerMachine = 100;
	};

	void SolveGPU(const GPUProblem& problem,
				  SolutionManager::GPUSolution& solution,
				  const SolverConfig& config = {});

	NeuralNetwork neuralNetwork;

	void PrintSchedule(const CPUSolution& solution, const JobShopData& data);

	// Add GPU kernel declaration
#ifdef __CUDACC__
	__global__ void SolveProblemsGPU(
		const JobShopData* problems,
		const float* weights,
		const float* biases,
		const int* topology,
		int num_layers,
		GPUSolution* solutions);
#endif

private:
	// JobShopHeuristic(NeuralNetwork&& net) : neuralNetwork(std::move(net)) {}

	static NeuralNetwork InitializeNetworkFromFile(const std::string& filename);

	__device__ void ExtractFeaturesGPU(
		const JobShopData& data,
		const Job& job,
		int opType,
		int machineId,
		int* machineAvailableTimes,	 // [num_machines]
		float* featuresOut			 // Output array
	);

	std::vector<float> ExtractFeatures(const JobShopData& data,
									   const Job& job,
									   const int& operationType,
									   const int& machineId,
									   const int& startTime,
									   const int& machineAvailableTime) const;

	void UpdateSchedule(JobShopData& data, int jobId, int operationId, int machineId, CPUSolution& solution);
};

#endif	// JOB_SHOP_HEURISTIC_CUH