#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>

#include "JobShopData.cuh"
#include "JobShopHeuristic.cuh"

float MeasureKernelTime(const std::function<void()>& kernelLaunch);
int main() {
    srand(time(0));

    const int numProblems = 9600;
    const std::vector<int> topology = {4, 32, 16, 1};

    try {
        // 1. Load all problems from JSON file (efficient batch load)
        std::vector<JobShopData> all_problems = JobShopData::LoadFromParallelJson("test_10k.json", numProblems);

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

        // 4. Allocate GPU memory for solutions
        auto solutions_batch = SolutionManager::CreateGPUSolutions(numProblems, all_problems[0].numMachines, 100);

        // 5. Create heuristic solver
        JobShopHeuristic heuristic(std::move(nn));

        // 6. Solve on GPU and measure time
        float kernelMs = MeasureKernelTime([&]{
            heuristic.SolveBatch(d_problems, &solutions_batch, numProblems);
        });
        std::cout << "Kernel execution time: " << kernelMs << " ms" << std::endl;

        cudaError_t kernelErr = cudaGetLastError();
        if(kernelErr != cudaSuccess) {
            std::cerr << "Kernel error: " << cudaGetErrorString(kernelErr) << "\n";
        }

        // 7. Download results from GPU
        std::vector<JobShopHeuristic::CPUSolution> solutions(numProblems);
        for(int i = 0; i < numProblems; ++i) {
            solutions[i].FromGPU(solutions_batch, i);
        }

        // 8. Print schedule for the first problem
        heuristic.PrintSchedule(solutions[0], all_problems[0]);

        // 9. Clean up GPU memory
        SolutionManager::FreeGPUSolutions(solutions_batch);
        JobShopDataGPU::FreeBatchGPUData(d_problems, d_jobs, d_ops, d_eligible, d_succ, d_procTimes);

    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

// Utility: measure kernel execution time in ms
float MeasureKernelTime(const std::function<void()>& kernelLaunch) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernelLaunch();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}
