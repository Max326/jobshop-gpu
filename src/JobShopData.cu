#include <cuda_runtime.h>

#include "JobShopData.cuh"

GPUProblem JobShopDataGPU::UploadToGPU(const JobShopData& problem) {
	GPUProblem gpuProblem;

	// 1. Copy basic info
	gpuProblem.numMachines = problem.numMachines;
	gpuProblem.numJobs = problem.numJobs;
	gpuProblem.numOpTypes = problem.numOpTypes;

	// 2. Process jobs (using pinned memory for better performance)
	std::vector<GPUJob> hostJobs(problem.numJobs);
	cudaMalloc(&gpuProblem.jobs, sizeof(GPUJob) * problem.numJobs);

	for(int j = 0; j < problem.numJobs; j++) {
		const auto& cpuJob = problem.jobs[j];
		GPUJob& gpuJob = hostJobs[j];

		gpuJob.id = cpuJob.id;
		// gpuJob.nextOpIndex = cpuJob.nextOpIndex;
		// gpuJob.lastOpEndTime = cpuJob.lastOpEndTime;
		gpuJob.operationCount = cpuJob.operations.size();

		// Allocate operations
		cudaMalloc(&gpuJob.operations, sizeof(GPUOperation) * gpuJob.operationCount);
		std::vector<GPUOperation> hostOps(gpuJob.operationCount);

		for(int o = 0; o < gpuJob.operationCount; o++) {
			const auto& cpuOp = cpuJob.operations[o];
			GPUOperation& gpuOp = hostOps[o];

			gpuOp.type = cpuOp.type;
			gpuOp.eligibleCount = cpuOp.eligibleMachines.size();

			// Allocate and copy eligible machines
			cudaMalloc(&gpuOp.eligibleMachines, sizeof(int) * gpuOp.eligibleCount);
			cudaMemcpy(gpuOp.eligibleMachines, cpuOp.eligibleMachines.data(),
					   sizeof(int) * gpuOp.eligibleCount, cudaMemcpyHostToDevice);
		}

		// Copy operations to device
		cudaMemcpy(gpuJob.operations, hostOps.data(),
				   sizeof(GPUOperation) * gpuJob.operationCount, cudaMemcpyHostToDevice);
	}

	// CRITICAL: Copy jobs array to device
	cudaMemcpy(gpuProblem.jobs, hostJobs.data(),
			   sizeof(GPUJob) * problem.numJobs, cudaMemcpyHostToDevice);

	// 3. Process processing times
	std::vector<int> flatTimes(problem.numOpTypes * problem.numMachines);
	for(int o = 0; o < problem.numOpTypes; o++) {
		for(int m = 0; m < problem.numMachines; m++) {
			flatTimes[o * problem.numMachines + m] = problem.processingTimes[o][m];
		}
	}
	cudaMalloc(&gpuProblem.processingTimes, sizeof(int) * flatTimes.size());
	cudaMemcpy(gpuProblem.processingTimes, flatTimes.data(),
			   sizeof(int) * flatTimes.size(), cudaMemcpyHostToDevice);

	// Error checking
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess) {
		FreeGPUData(gpuProblem);  // Clean up if error
		throw std::runtime_error("CUDA error during upload: " +
								 std::string(cudaGetErrorString(err)));
	}

	return gpuProblem;
}

void JobShopDataGPU::FreeGPUData(GPUProblem& gpuProblem) {
	// Helper function to free nested structures
	auto FreeJob = [](GPUJob& job) {
		if(job.operations) {
			std::vector<GPUOperation> tempOps(job.operationCount);
			cudaMemcpy(tempOps.data(), job.operations,
					   sizeof(GPUOperation) * job.operationCount, cudaMemcpyDeviceToHost);

			for(auto& op: tempOps) {
				if(op.eligibleMachines) {
					cudaFree(op.eligibleMachines);
				}
			}
			cudaFree(job.operations);
		}
	};

	// 1. Free jobs and their nested data
	if(gpuProblem.jobs) {
		std::vector<GPUJob> tempJobs(gpuProblem.numJobs);
		cudaMemcpy(tempJobs.data(), gpuProblem.jobs,
				   sizeof(GPUJob) * gpuProblem.numJobs, cudaMemcpyDeviceToHost);

		for(auto& job: tempJobs) {
			FreeJob(job);
		}
		cudaFree(gpuProblem.jobs);
	}

	// 2. Free processing times
	if(gpuProblem.processingTimes) {
		cudaFree(gpuProblem.processingTimes);
	}

	// 3. Reset struct
	gpuProblem = GPUProblem {};
}