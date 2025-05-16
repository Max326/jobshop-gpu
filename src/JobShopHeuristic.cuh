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

#define MAX_MACHINES 5	 // TODO: make this dynamic
#define MAX_JOB_TYPES 15
#define MAX_JOBS	 30
#define MAX_OPS		 15
#define MAX_OP_TYPES 15

// Structure for scheduled operation
struct OperationSchedule {
    int jobId;
    int opType;
    int startTime;
    int endTime;
};

// GPU solutions manager
class SolutionManager
{
public:
    struct GPUSolutions {
        OperationSchedule* allSchedules;  // [machine][operation]
        int* allScheduleCounts;			  // Operations per machine
        int* allMakespans;
        int numProblems;
        int numMachines;
        int maxOps;
    };

    static GPUSolutions CreateGPUSolutions(int numProblems, int numMachines, int maxOpsPerMachine);
    static void FreeGPUSolutions(GPUSolutions& solution);
};

// Heuristic solver for Job Shop
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

    JobShopHeuristic(const std::vector<int>& topology);
    JobShopHeuristic(const std::string& filename);
    JobShopHeuristic(NeuralNetwork&& net);

    // GPU Interface
    struct SolverConfig {
        int maxThreadsPerBlock = 256;
        int maxOperationsPerMachine = 100;
    };

	static void SolveBatchNew(
		const GPUProblem* problems,
		const NeuralNetwork::DeviceEvaluator* evaluators,
        GPUOperation* ops_working,
		float* results,
		int numProblems,
		int numWeights,
        int maxOpsPerProblem
    );

    void PrintSchedule(const CPUSolution& solution, JobShopData data);

public:
    NeuralNetwork neuralNetwork;

private:
    static NeuralNetwork InitializeNetworkFromFile(const std::string& filename);

    void UpdateSchedule(JobShopData& data, int jobId, int operationId, int machineId, CPUSolution& solution);
};

__global__ void SolveFJSSPKernel(
    const GPUProblem* problems,
    const NeuralNetwork::DeviceEvaluator nn_eval,
    SolutionManager::GPUSolutions* solutions,
    int total_problems);

__global__ void SolveManyWeightsKernel(
    const GPUProblem* problems,
    const NeuralNetwork::DeviceEvaluator* evaluators,
    GPUOperation* ops_working,
    float* results,
    int numProblems,
    int maxOpsPerProblem
);

__device__ void PrintProblemDetails(const GPUProblem& problem);

#endif	// JOB_SHOP_HEURISTIC_CUH