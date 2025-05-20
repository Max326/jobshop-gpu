#include <cuda_runtime.h>
#include "JobShopData.cuh"

// Prepare batch data on CPU for GPU upload
BatchJobShopGPUData JobShopDataGPU::PrepareBatchCPU(const std::vector<JobShopData>& problems) {
    BatchJobShopGPUData batch;
    batch.numProblems = problems.size();

    // Initialize offsets
    batch.jobsOffsets.push_back(0);
    batch.operationsOffsets.push_back(0);
    batch.eligibleOffsets.push_back(0);
    batch.successorsOffsets.push_back(0);
    batch.processingTimesOffsets.push_back(0);

    for (const auto& problem : problems) {
        int current_problem_total_ops = 0; // Licznik operacji dla bieżącego problemu

        // Jobs
        for (const auto& job : problem.jobs) {
            GPUJob gpuJob;
            gpuJob.id = job.id;
            gpuJob.type = job.type;
            gpuJob.operationsOffset = batch.operations.size(); // To jest offset w globalnym buforze operacji batcha
            gpuJob.operationCount = job.operations.size();
            batch.jobs.push_back(gpuJob);

            current_problem_total_ops += job.operations.size(); // Sumuj operacje dla problemu

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

        // Processing times (flattened)
        for (const auto& row : problem.processingTimes)
            for (int t : row)
                batch.processingTimes.push_back(t);

        // Offsets for this problem
        batch.jobsOffsets.push_back(batch.jobs.size());
        batch.operationsOffsets.push_back(batch.operations.size());
        batch.eligibleOffsets.push_back(batch.eligibleMachines.size());
        batch.successorsOffsets.push_back(batch.successorsIDs.size());
        batch.processingTimesOffsets.push_back(batch.processingTimes.size());

        // GPUProblem struct (pointers set later)
        GPUProblem gpuProblem;
        gpuProblem.numMachines = problem.numMachines;
        gpuProblem.numJobs = problem.numJobs;
        gpuProblem.numOpTypes = problem.numOpTypes;
        gpuProblem.totalOpsCount = current_problem_total_ops; // <<< USTAW POLE totalOpsCount
        gpuProblem.jobs = nullptr;
        gpuProblem.operations = nullptr;
        gpuProblem.eligibleMachines = nullptr;
        gpuProblem.successorsIDs = nullptr;
        gpuProblem.processingTimes = nullptr;
        batch.gpuProblems.push_back(gpuProblem);
    }

    return batch;
}

// Upload batch data to GPU
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

    // Make offsets local for each problem
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

    // Copy data to GPU
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

    // Prepare GPUProblem with device pointers
    // Create a temporary host-side vector to modify before copying to device
    std::vector<GPUProblem> host_gpuProblems = batch.gpuProblems; 
    for (int i = 0; i < numProblems; ++i) {
        host_gpuProblems[i].jobs = d_jobs + batch.jobsOffsets[i];
        host_gpuProblems[i].operations = d_ops + batch.operationsOffsets[i]; // This points to the template operations for this problem
        host_gpuProblems[i].eligibleMachines = d_eligible + batch.eligibleOffsets[i];
        host_gpuProblems[i].successorsIDs = d_succ + batch.successorsOffsets[i];
        host_gpuProblems[i].processingTimes = d_procTimes + batch.processingTimesOffsets[i];
        // host_gpuProblems[i].totalOpsCount is already set in PrepareBatchCPU
    }

    cudaMalloc(&d_gpuProblems, numProblems * sizeof(GPUProblem));
    cudaMemcpy(d_gpuProblems, host_gpuProblems.data(), numProblems * sizeof(GPUProblem), cudaMemcpyHostToDevice);
}

// Free GPU memory
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

// Download from GPU (not implemented)
void JobShopDataGPU::DownloadFromGPU(GPUProblem&, JobShopData&) {
    // Not implemented
}