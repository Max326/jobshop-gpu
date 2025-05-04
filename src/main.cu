#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "JobShopData.cuh"
#include "JobShopHeuristic.cuh"

int main() {
    srand(time(0));

    const int numProblems = 50;
    std::vector<JobShopData> all_problems;
    const std::vector<int> topology = {4, 32, 16, 1};

    try {
        // 1. Load problem data
        for (int i = 0; i < numProblems; ++i) {
            JobShopData data;
            data.LoadFromParallelJson("test_100.json", i);
            all_problems.push_back(std::move(data));
        }

        // 2. Load neural network
        NeuralNetwork nn;
        nn.LoadFromJson("weights_and_biases");

        // 3. Prepare batch and upload to GPU
        auto batch = JobShopDataGPU::PrepareBatchCPU(all_problems);

        GPUProblem* d_problems = nullptr;
        GPUJob* d_jobs = nullptr;
        GPUOperation* d_ops = nullptr;
        int* d_eligible = nullptr;
        int* d_succ = nullptr;
        int* d_procTimes = nullptr;
        int numProblemsGPU = 0;

        JobShopDataGPU::UploadBatchToGPU(
            batch, d_problems, d_jobs, d_ops, d_eligible, d_succ, d_procTimes, numProblemsGPU);

        auto solutions_batch = SolutionManager::CreateGPUSolutions(numProblems, all_problems[0].numMachines, 100);

        // 4. Create heuristic solver
        JobShopHeuristic heuristic(std::move(nn));

        // 5. Solve on GPU
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        heuristic.SolveBatch(d_problems, &solutions_batch, numProblems);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaError_t kernelErr = cudaGetLastError();
        if(kernelErr != cudaSuccess) {
            std::cerr << "Kernel error: " << cudaGetErrorString(kernelErr) << "\n";
        }

        // 6. Download results
        std::vector<JobShopHeuristic::CPUSolution> solutions(numProblems);
        for(int i = 0; i < numProblems; ++i) {
            solutions[i].FromGPU(solutions_batch, i);
        }

        heuristic.PrintSchedule(solutions[0], all_problems[0]);

        // 7. Clean up GPU memory
        SolutionManager::FreeGPUSolutions(solutions_batch);
        JobShopDataGPU::FreeBatchGPUData(d_problems, d_jobs, d_ops, d_eligible, d_succ, d_procTimes);

    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}