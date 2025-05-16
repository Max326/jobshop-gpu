/* #include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>

#include "JobShopData.cuh"
#include "JobShopHeuristic.cuh"

float MeasureKernelTime(const std::function<void()>& kernelLaunch);
int main() {
    srand(time(0));

    const int numProblems = 50;
    const int numWeights = 192;
    const std::vector<int> topology = {4, 32, 16, 1};

    try {
        // 1. Load all problems from JSON file (efficient batch load)
        std::vector<JobShopData> all_problems = JobShopData::LoadFromParallelJson("test_10k.json", numProblems);

        // 2. Load neural network
        std::vector<NeuralNetwork> networks = NeuralNetwork::LoadBatchFromJson("weights_and_biases_192.json");
        std::vector<NeuralNetwork::DeviceEvaluator> evaluators;

        for (auto& net : networks) {
            evaluators.push_back(net.GetDeviceEvaluator());
        }

        NeuralNetwork::DeviceEvaluator* d_evaluators = nullptr;
        cudaMalloc(&d_evaluators, sizeof(NeuralNetwork::DeviceEvaluator) * numWeights);
        cudaMemcpy(d_evaluators, evaluators.data(), sizeof(NeuralNetwork::DeviceEvaluator) * numWeights, cudaMemcpyHostToDevice);
        
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

        int maxOpsPerProblem = 0;
        for (int p = 0; p < numProblems; ++p) {
            int opsCount = batch.operationsOffsets[p+1] - batch.operationsOffsets[p];
            if (opsCount > maxOpsPerProblem) maxOpsPerProblem = opsCount;
        }
        
        std::vector<GPUOperation> ops_working(numWeights * numProblems * maxOpsPerProblem);
        
        for (int w = 0; w < numWeights; ++w) {
            for (int p = 0; p < numProblems; ++p) {
                int base = (w * numProblems + p) * maxOpsPerProblem;
                int opsOffset = batch.operationsOffsets[p];
                int opsCount = batch.operationsOffsets[p+1] - batch.operationsOffsets[p];
                memcpy(&ops_working[base], &batch.operations[opsOffset], opsCount * sizeof(GPUOperation));
            }
        }
        
        GPUOperation* d_ops_working = nullptr;
        cudaMalloc(&d_ops_working, ops_working.size() * sizeof(GPUOperation));
        cudaMemcpy(d_ops_working, ops_working.data(), ops_working.size() * sizeof(GPUOperation), cudaMemcpyHostToDevice);
        
        // Allocate      
        float* d_results = nullptr;
        cudaMalloc(&d_results, sizeof(float) * numWeights);
        
        std::vector<float> results(numWeights, 0.0f);
        
        float kernelMs = MeasureKernelTime([&]{
            JobShopHeuristic::SolveBatchNew(
                d_problems, d_evaluators, d_ops_working, d_results, numProblems, numWeights, maxOpsPerProblem
            );
        });
        std::cout << "Kernel execution time: " << kernelMs << " ms" << std::endl;
        
        cudaError_t kernelErr = cudaGetLastError();
        if(kernelErr != cudaSuccess) {
            std::cerr << "Kernel error: " << cudaGetErrorString(kernelErr) << "\n";
        }
        
        cudaMemcpy(results.data(), d_results, sizeof(float) * numWeights, cudaMemcpyDeviceToHost);
        for(int i = 0; i < numWeights; ++i) {
            std::cout << "Avg makespan for weights set " << i << ": " << results[i] << std::endl;
        }
        
        JobShopDataGPU::FreeBatchGPUData(d_problems, d_jobs, d_ops, d_eligible, d_succ, d_procTimes);
        cudaFree(d_evaluators);
        cudaFree(d_results);
        cudaFree(d_ops_working);

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
 */