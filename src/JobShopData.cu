#include <cuda_runtime.h>

#include "JobShopData.cuh"

GPUProblem JobShopDataGPU::UploadToGPU(const JobShopData& problem) {
    GPUProblem gpuProblem;

    // 1. Copy basic info
    gpuProblem.numMachines = problem.numMachines;
    gpuProblem.numJobs = problem.numJobs;
    gpuProblem.numOpTypes = problem.numOpTypes;

    // 2. count the ops, machines and succesors 
    int totalOps = 0, totalEligible = 0, totalSuccessors = 0;
    for (const auto& job : problem.jobs) {
        totalOps += job.operations.size();
        for (const auto& op : job.operations) {
            totalEligible += op.eligibleMachines.size();
            totalSuccessors += op.successorsIDs.size();
        }
    }

    // 3. Allocate
    std::vector<GPUJob> hostJobs(problem.numJobs);
    std::vector<GPUOperation> allOps(totalOps);
    std::vector<int> allEligible(totalEligible);
    std::vector<int> allSuccessors(totalSuccessors);

    // 4. fill 
    int opOffset = 0, eligibleOffset = 0, succOffset = 0;
    for (int j = 0; j < problem.numJobs; ++j) {
        const auto& cpuJob = problem.jobs[j];
        GPUJob& gpuJob = hostJobs[j];
        gpuJob.id = cpuJob.id;
        gpuJob.operationsOffset = opOffset;
        gpuJob.operationCount = cpuJob.operations.size();

        for (size_t o = 0; o < cpuJob.operations.size(); ++o) {
            const auto& cpuOp = cpuJob.operations[o];
            GPUOperation& gpuOp = allOps[opOffset];
            gpuOp.type = cpuOp.type;
            gpuOp.predecessorCount = cpuOp.predecessorCount;
            gpuOp.lastPredecessorEndTime = cpuOp.lastPredecessorEndTime;

            gpuOp.eligibleMachinesOffset = eligibleOffset;
            gpuOp.eligibleCount = cpuOp.eligibleMachines.size();
            for (size_t em = 0; em < cpuOp.eligibleMachines.size(); ++em)
                allEligible[eligibleOffset + em] = cpuOp.eligibleMachines[em];
            eligibleOffset += cpuOp.eligibleMachines.size();

            gpuOp.successorsOffset = succOffset;
            gpuOp.successorCount = cpuOp.successorsIDs.size();
            for (size_t s = 0; s < cpuOp.successorsIDs.size(); ++s)
                allSuccessors[succOffset + s] = cpuOp.successorsIDs[s];
            succOffset += cpuOp.successorsIDs.size();

            opOffset++;
        }
    }

    // 5. cudaMalloc/cudaMemcpy
    cudaMalloc(&gpuProblem.jobs, sizeof(GPUJob) * problem.numJobs);
    cudaMemcpy(gpuProblem.jobs, hostJobs.data(), sizeof(GPUJob) * problem.numJobs, cudaMemcpyHostToDevice);

    cudaMalloc(&gpuProblem.operations, sizeof(GPUOperation) * totalOps);
    cudaMemcpy(gpuProblem.operations, allOps.data(), sizeof(GPUOperation) * totalOps, cudaMemcpyHostToDevice);

    cudaMalloc(&gpuProblem.eligibleMachines, sizeof(int) * totalEligible);
    cudaMemcpy(gpuProblem.eligibleMachines, allEligible.data(), sizeof(int) * totalEligible, cudaMemcpyHostToDevice);

    cudaMalloc(&gpuProblem.successorsIDs, sizeof(int) * totalSuccessors);
    cudaMemcpy(gpuProblem.successorsIDs, allSuccessors.data(), sizeof(int) * totalSuccessors, cudaMemcpyHostToDevice);

    // 6. Processing times 
    std::vector<int> flatTimes(problem.numOpTypes * problem.numMachines);
    for(int o = 0; o < problem.numOpTypes; o++) {
        for(int m = 0; m < problem.numMachines; m++) {
            flatTimes[o * problem.numMachines + m] = problem.processingTimes[o][m];
        }
    }
    cudaMalloc(&gpuProblem.processingTimes, sizeof(int) * flatTimes.size());
    cudaMemcpy(gpuProblem.processingTimes, flatTimes.data(),
               sizeof(int) * flatTimes.size(), cudaMemcpyHostToDevice);

    return gpuProblem;
}

GPUProblem JobShopDataGPU::UploadParallelToGPU(const JobShopData& problem) {
    return UploadToGPU(problem);
}

void JobShopDataGPU::FreeGPUData(GPUProblem& gpuProblem) {
    if(gpuProblem.jobs) cudaFree(gpuProblem.jobs);
    if(gpuProblem.operations) cudaFree(gpuProblem.operations);
    if(gpuProblem.eligibleMachines) cudaFree(gpuProblem.eligibleMachines);
    if(gpuProblem.successorsIDs) cudaFree(gpuProblem.successorsIDs);
    if(gpuProblem.processingTimes) cudaFree(gpuProblem.processingTimes);
    gpuProblem = GPUProblem {};
}