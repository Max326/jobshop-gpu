#include <cuda_runtime.h>

#include "JobShopData.cuh"

BatchJobShopGPUData JobShopDataGPU::PrepareBatchCPU(const std::vector<JobShopData>& problems) {
    BatchJobShopGPUData batch;
    batch.numProblems = problems.size();

    int jobsOffset = 0, opsOffset = 0, eligibleOffset = 0, succOffset = 0, procTimesOffset = 0;

    for (const auto& problem : problems) {
        batch.jobsOffsets.push_back(batch.jobs.size());
        batch.operationsOffsets.push_back(batch.operations.size());
        batch.eligibleOffsets.push_back(batch.eligibleMachines.size());
        batch.successorsOffsets.push_back(batch.successorsIDs.size());
        batch.processingTimesOffsets.push_back(batch.processingTimes.size());

        // Jobs
        for (const auto& job : problem.jobs) {
            GPUJob gpuJob;
            gpuJob.id = job.id;
            gpuJob.operationsOffset = batch.operations.size();
            gpuJob.operationCount = job.operations.size();
            batch.jobs.push_back(gpuJob);

            // Operations
            for (const auto& op : job.operations) {
                GPUOperation gpuOp;
                gpuOp.type = op.type;
                gpuOp.predecessorCount = op.predecessorCount;
                gpuOp.lastPredecessorEndTime = op.lastPredecessorEndTime;
                gpuOp.eligibleMachinesOffset = batch.eligibleMachines.size();
                gpuOp.eligibleCount = op.eligibleMachines.size();
                for (int m : op.eligibleMachines)
                    batch.eligibleMachines.push_back(m);
                gpuOp.successorsOffset = batch.successorsIDs.size();
                gpuOp.successorCount = op.successorsIDs.size();
                for (int s : op.successorsIDs)
                    batch.successorsIDs.push_back(s);
                batch.operations.push_back(gpuOp);
            }
        }

        // Processing times (spłaszczone)
        for (const auto& row : problem.processingTimes)
            for (int t : row)
                batch.processingTimes.push_back(t);

        // GPUProblem z offsetami (nie wskaźnikami!)
        GPUProblem gpuProblem;
        gpuProblem.numMachines = problem.numMachines;
        gpuProblem.numJobs = problem.numJobs;
        gpuProblem.numOpTypes = problem.numOpTypes;
        // Wskaźniki ustaw na nullptr, offsety będą używane po stronie GPU
        gpuProblem.jobs = nullptr;
        gpuProblem.operations = nullptr;
        gpuProblem.eligibleMachines = nullptr;
        gpuProblem.successorsIDs = nullptr;
        gpuProblem.processingTimes = nullptr;
        batch.gpuProblems.push_back(gpuProblem);
    }

    // Dodaj końcowe offsety
    batch.jobsOffsets.push_back(batch.jobs.size());
    batch.operationsOffsets.push_back(batch.operations.size());
    batch.eligibleOffsets.push_back(batch.eligibleMachines.size());
    batch.successorsOffsets.push_back(batch.successorsIDs.size());
    batch.processingTimesOffsets.push_back(batch.processingTimes.size());

    return batch;
}
void JobShopDataGPU::UploadBatchToGPU(
    BatchJobShopGPUData& batch,
    GPUProblem*& d_gpuProblems,
    GPUJob*& d_jobs,
    GPUOperation*& d_ops,
    int*& d_eligible,
    int*& d_succ,
    int*& d_procTimes,
    int& numProblems)
{
    numProblems = batch.numProblems;

    // calculate offsets
    for (int i = 0; i < numProblems; ++i) {
        int opBase = batch.operationsOffsets[i];
        int eligibleBase = batch.eligibleOffsets[i];
        int succBase = batch.successorsOffsets[i];

        for(int j = batch.jobsOffsets[i]; j < batch.jobsOffsets[i+1]; ++j) {
            batch.jobs[j].operationsOffset -= opBase;
        }
        for(int o = batch.operationsOffsets[i]; o < batch.operationsOffsets[i+1]; ++o) {
            batch.operations[o].eligibleMachinesOffset -= eligibleBase;
            batch.operations[o].successorsOffset -= succBase;
        }
    }

    // copy to GPU
    cudaMalloc(&d_jobs, batch.jobs.size() * sizeof(GPUJob));
    cudaMemcpy(d_jobs, batch.jobs.data(), batch.jobs.size() * sizeof(GPUJob), cudaMemcpyHostToDevice);

    cudaMalloc(&d_ops, batch.operations.size() * sizeof(GPUOperation));
    cudaMemcpy(d_ops, batch.operations.data(), batch.operations.size() * sizeof(GPUOperation), cudaMemcpyHostToDevice);

    cudaMalloc(&d_eligible, batch.eligibleMachines.size() * sizeof(int));
    cudaMemcpy(d_eligible, batch.eligibleMachines.data(), batch.eligibleMachines.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_succ, batch.successorsIDs.size() * sizeof(int));
    cudaMemcpy(d_succ, batch.successorsIDs.data(), batch.successorsIDs.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_procTimes, batch.processingTimes.size() * sizeof(int));
    cudaMemcpy(d_procTimes, batch.processingTimes.data(), batch.processingTimes.size() * sizeof(int), cudaMemcpyHostToDevice);

    // 3. Prepare GPUProblem
    std::vector<GPUProblem> gpuProblems = batch.gpuProblems;
    for (int i = 0; i < numProblems; ++i) {
        gpuProblems[i].jobs = d_jobs + batch.jobsOffsets[i];
        gpuProblems[i].operations = d_ops + batch.operationsOffsets[i];
        gpuProblems[i].eligibleMachines = d_eligible + batch.eligibleOffsets[i];
        gpuProblems[i].successorsIDs = d_succ + batch.successorsOffsets[i];
        gpuProblems[i].processingTimes = d_procTimes + batch.processingTimesOffsets[i];
    }

    cudaMalloc(&d_gpuProblems, numProblems * sizeof(GPUProblem));
    cudaMemcpy(d_gpuProblems, gpuProblems.data(), numProblems * sizeof(GPUProblem), cudaMemcpyHostToDevice);
}

void JobShopDataGPU::FreeBatchGPUData(GPUProblem* d_gpuProblems, 
                                      GPUJob* d_jobs, GPUOperation* d_ops, 
                                      int* d_eligible, int* d_succ, int* d_procTimes) {
    if(d_gpuProblems) cudaFree(d_gpuProblems);
    if(d_jobs) cudaFree(d_jobs);
    if(d_ops) cudaFree(d_ops);
    if(d_eligible) cudaFree(d_eligible);
    if(d_succ) cudaFree(d_succ);
    if(d_procTimes) cudaFree(d_procTimes);
}