#include <cuda_runtime.h>

#include "JobShopData.cuh"

GPUProblem JobShopDataGPU::UploadToGPU(const JobShopData& problem) {
	GPUProblem gpuProblem;

	// 1. Allocate and copy basic info
	gpuProblem.numMachines = problem.numMachines;
	gpuProblem.numJobs = problem.numJobs;
	gpuProblem.numOpTypes = problem.numOpTypes;

	// 2. Debug print before upload
	std::cout << "Uploading problem with:\n";
	std::cout << "  Machines: " << problem.numMachines << "\n";
	std::cout << "  Jobs: " << problem.numJobs << "\n";
	std::cout << "  First job operations: " << problem.jobs[0].operations.size() << "\n";

	// 2. Process jobs
	cudaMalloc(&gpuProblem.jobs, sizeof(GPUJob) * problem.numJobs);
	std::vector<GPUJob> tempJobs(problem.numJobs);

	for(int j = 0; j < problem.numJobs; j++) {
		const auto& cpuJob = problem.jobs[j];
		GPUJob gpuJob;

		gpuJob.id = cpuJob.id;
		gpuJob.nextOpIndex = cpuJob.nextOpIndex;
		gpuJob.lastOpEndTime = cpuJob.lastOpEndTime;
		gpuJob.operationCount = cpuJob.operations.size();

		// Allocate operations
		cudaMalloc(&gpuJob.operations, sizeof(GPUOperation) * gpuJob.operationCount);
		std::vector<GPUOperation> tempOps(gpuJob.operationCount);

		for(int o = 0; o < gpuJob.operationCount; o++) {
			const auto& cpuOp = cpuJob.operations[o];
			GPUOperation gpuOp;

			gpuOp.type = cpuOp.type;
			gpuOp.eligibleCount = cpuOp.eligibleMachines.size();

			// Allocate eligible machines
			cudaMalloc(&gpuOp.eligibleMachines, sizeof(int) * gpuOp.eligibleCount);
			cudaMemcpy(gpuOp.eligibleMachines, cpuOp.eligibleMachines.data(),
					   sizeof(int) * gpuOp.eligibleCount, cudaMemcpyHostToDevice);

			tempOps[o] = gpuOp;
		}

		// Copy operations to device
		cudaMemcpy(gpuJob.operations, tempOps.data(),
				   sizeof(GPUOperation) * gpuJob.operationCount, cudaMemcpyHostToDevice);

		tempJobs[j] = gpuJob;
	}

	// 3. Process processing times (flatten 2D array)
	int processingSize = problem.numOpTypes * problem.numMachines;
	cudaMalloc(&gpuProblem.processingTimes, sizeof(int) * processingSize);

	std::vector<int> flatTimes(processingSize);
	for(int o = 0; o < problem.numOpTypes; o++) {
		for(int m = 0; m < problem.numMachines; m++) {
			flatTimes[o * problem.numMachines + m] = problem.processingTimes[o][m];
		}
	}
	cudaMemcpy(gpuProblem.processingTimes, flatTimes.data(),
			   sizeof(int) * processingSize, cudaMemcpyHostToDevice);

	return gpuProblem;
}

void JobShopDataGPU::DownloadFromGPU(GPUProblem& gpuProblem, JobShopData& cpuProblem) {
	// 1. Download basic info
	cpuProblem.numMachines = gpuProblem.numMachines;
	cpuProblem.numJobs = gpuProblem.numJobs;
	cpuProblem.numOpTypes = gpuProblem.numOpTypes;

	// 2. Download jobs
	std::vector<GPUJob> tempJobs(gpuProblem.numJobs);
	cudaMemcpy(tempJobs.data(), gpuProblem.jobs,
			   sizeof(GPUJob) * gpuProblem.numJobs, cudaMemcpyDeviceToHost);

	cpuProblem.jobs.resize(gpuProblem.numJobs);
	for(int j = 0; j < gpuProblem.numJobs; j++) {
		GPUJob& gpuJob = tempJobs[j];
		Job& cpuJob = cpuProblem.jobs[j];

		cpuJob.id = gpuJob.id;
		cpuJob.nextOpIndex = gpuJob.nextOpIndex;
		cpuJob.lastOpEndTime = gpuJob.lastOpEndTime;

		// Download operations
		std::vector<GPUOperation> tempOps(gpuJob.operationCount);
		cudaMemcpy(tempOps.data(), gpuJob.operations,
				   sizeof(GPUOperation) * gpuJob.operationCount, cudaMemcpyDeviceToHost);

		cpuJob.operations.resize(gpuJob.operationCount);
		for(int o = 0; o < gpuJob.operationCount; o++) {
			GPUOperation& gpuOp = tempOps[o];
			Operation& cpuOp = cpuJob.operations[o];

			cpuOp.type = gpuOp.type;

			// Download eligible machines
			std::vector<int> tempMachines(gpuOp.eligibleCount);
			cudaMemcpy(tempMachines.data(), gpuOp.eligibleMachines,
					   sizeof(int) * gpuOp.eligibleCount, cudaMemcpyDeviceToHost);
			cpuOp.eligibleMachines = tempMachines;
		}
	}

	// 3. Download processing times
	std::vector<int> flatTimes(gpuProblem.numOpTypes * gpuProblem.numMachines);
	cudaMemcpy(flatTimes.data(), gpuProblem.processingTimes,
			   sizeof(int) * flatTimes.size(), cudaMemcpyDeviceToHost);

	cpuProblem.processingTimes.resize(gpuProblem.numOpTypes);
	for(int o = 0; o < gpuProblem.numOpTypes; o++) {
		cpuProblem.processingTimes[o].resize(gpuProblem.numMachines);
		for(int m = 0; m < gpuProblem.numMachines; m++) {
			cpuProblem.processingTimes[o][m] =
				flatTimes[o * gpuProblem.numMachines + m];
		}
	}
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