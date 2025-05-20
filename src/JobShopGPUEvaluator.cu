#include "JobShopGPUEvaluator.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <chrono>

JobShopGPUEvaluator::JobShopGPUEvaluator(const std::string& problem_file, const std::vector<int>& nn_topology, const int &population_size)
    : nn_topology_(nn_topology)
{
    // all problems at once 
    cpu_problems_ = JobShopData::LoadFromParallelJson(problem_file, 400);//TODO fix nummber of problem assignment 
    if (cpu_problems_.empty())
        throw std::runtime_error("No problems loaded!");

    
    max_ops_per_problem_ = 0;
    for (const auto& prob : cpu_problems_) {
        int ops = 0;
        for (const auto& job : prob.jobs)
            ops += job.operations.size();
        if (ops > max_ops_per_problem_) max_ops_per_problem_ = ops;
    }

    nn_total_params_ = NeuralNetwork::CalculateTotalParameters(nn_topology_);

    // Initialize DeviceEvaluator pool:
    nn_candidate_count_ = population_size; // number of candidates, use the value you are using in the CMAES
    neural_networks_.resize(nn_candidate_count_);
    host_evaluators_.resize(nn_candidate_count_);

    for (int r = 0; r < nn_candidate_count_; ++r) {
        neural_networks_[r] = NeuralNetwork(nn_topology_); // Create a NeuralNetwork
        host_evaluators_[r] = neural_networks_[r].GetDeviceEvaluator(); // Get its DeviceEvaluator
    }

    // Allocate and copy DeviceEvaluators to GPU
    cudaMalloc(&d_evaluators_, sizeof(NeuralNetwork::DeviceEvaluator) * nn_candidate_count_);
    cudaMemcpy(d_evaluators_, host_evaluators_.data(), sizeof(NeuralNetwork::DeviceEvaluator) * nn_candidate_count_, cudaMemcpyHostToDevice);
}

JobShopGPUEvaluator::~JobShopGPUEvaluator() {
    FreeProblemDataGPU();
    cudaFree(d_evaluators_);
}

void JobShopGPUEvaluator::FreeProblemDataGPU() {
    JobShopDataGPU::FreeBatchGPUData(d_problems_, d_jobs_, d_ops_, d_eligible_, d_succ_, d_procTimes_);
    d_problems_ = nullptr;
    d_jobs_ = nullptr;
    d_ops_ = nullptr;
    d_eligible_ = nullptr;
    d_succ_ = nullptr;
    d_procTimes_ = nullptr;
}

void JobShopGPUEvaluator::PrepareProblemDataGPU(const std::vector<JobShopData>& batch) {
    FreeProblemDataGPU();
    cpu_batch_data_ = JobShopDataGPU::PrepareBatchCPU(batch);
    num_problems_to_evaluate_ = batch.size();

    int num_problems_on_gpu = 0;
    JobShopDataGPU::UploadBatchToGPU(
        cpu_batch_data_, d_problems_, d_jobs_, d_ops_, 
        d_eligible_, d_succ_, d_procTimes_, num_problems_on_gpu
    );
    if (num_problems_on_gpu != num_problems_to_evaluate_)
        throw std::runtime_error("Mismatch in number of problems uploaded to GPU.");
}

bool JobShopGPUEvaluator::SetCurrentBatch(int batch_start, int batch_size) {
    auto t0 = std::chrono::high_resolution_clock::now();
    if (batch_start >= (int)cpu_problems_.size())
        return false;
    int batch_end = std::min(batch_start + batch_size, (int)cpu_problems_.size());
    std::vector<JobShopData> batch(cpu_problems_.begin() + batch_start, cpu_problems_.begin() + batch_end);
    auto t1 = std::chrono::high_resolution_clock::now();
    PrepareProblemDataGPU(batch);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMER][CPU] Batch slicing: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms, "
              << "PrepareProblemDataGPU: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms"
              << std::endl;
    return true;
}

Eigen::VectorXd JobShopGPUEvaluator::EvaluateCandidates(const Eigen::MatrixXd& candidates) {
    auto t0 = std::chrono::high_resolution_clock::now();

    int nn_candidate_count = candidates.cols();
    if (candidates.rows() != nn_total_params_)
        throw std::runtime_error("Mismatch in number of weights per NN candidate.");

    // Update weights in existing NeuralNetwork objects
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < nn_candidate_count; ++r) {
        int paramIdx = 0;
        std::vector<std::vector<float>> weights_for_net; // Renamed to avoid conflict
        std::vector<std::vector<float>> biases_for_net; // Renamed to avoid conflict

        for (size_t i = 1; i < nn_topology_.size(); ++i) {
            int prevLayerSize = nn_topology_[i - 1];
            int currLayerSize = nn_topology_[i];

            std::vector<float> layerWeights(prevLayerSize * currLayerSize);
            std::vector<float> layerBiases(currLayerSize);

            for (int w = 0; w < prevLayerSize * currLayerSize; ++w)
                layerWeights[w] = static_cast<float>(candidates(paramIdx++, r));
            weights_for_net.push_back(layerWeights);

            for (int b = 0; b < currLayerSize; ++b)
                layerBiases[b] = static_cast<float>(candidates(paramIdx++, r));
            biases_for_net.push_back(layerBiases);
        }
      
        // set the new weights to the pre-existing neural network, be careful that the new sizes are correct
        neural_networks_[r].weights = weights_for_net;
        neural_networks_[r].biases = biases_for_net;
        neural_networks_[r].FlattenParams();

        // copy weights and biases to GPU
        size_t weight_offset = 0;
        size_t bias_offset = 0;
        for (size_t i = 0; i < weights_for_net.size(); ++i) {
          cudaMemcpy(neural_networks_[r].cudaData->d_weights + weight_offset,
                     weights_for_net[i].data(),
                     weights_for_net[i].size() * sizeof(float),
                     cudaMemcpyHostToDevice);
          cudaMemcpy(neural_networks_[r].cudaData->d_biases + bias_offset,
                     biases_for_net[i].data(),
                     biases_for_net[i].size() * sizeof(float),
                     cudaMemcpyHostToDevice);
          weight_offset += weights_for_net[i].size();
          bias_offset += biases_for_net[i].size();
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    // DeviceEvaluator H2D: Use the pre-allocated d_evaluators_
    auto t3 = std::chrono::high_resolution_clock::now();

    // Rest of your code remains mostly the same, using d_evaluators_
    std::vector<GPUOperation> ops_working(nn_candidate_count * num_problems_to_evaluate_ * max_ops_per_problem_);
    for (int w = 0; w < nn_candidate_count; ++w) {
        for (int p = 0; p < num_problems_to_evaluate_; ++p) {
            int base = (w * num_problems_to_evaluate_ + p) * max_ops_per_problem_;
            int opsOffset = cpu_batch_data_.operationsOffsets[p];
            int opsCount = cpu_batch_data_.operationsOffsets[p + 1] - cpu_batch_data_.operationsOffsets[p];
            memcpy(&ops_working[base], &cpu_batch_data_.operations[opsOffset], opsCount * sizeof(GPUOperation));
        }
    }

    auto t4 = std::chrono::high_resolution_clock::now();

    GPUOperation* d_ops_working = nullptr;
    cudaMalloc(&d_ops_working, ops_working.size() * sizeof(GPUOperation));
    cudaMemcpy(d_ops_working, ops_working.data(), ops_working.size() * sizeof(GPUOperation), cudaMemcpyHostToDevice);

    float* d_results = nullptr;
    cudaMalloc(&d_results, sizeof(float) * nn_candidate_count);

    // Kernel
    auto t5 = std::chrono::high_resolution_clock::now();
    cudaStream_t stream; // I don't know if this helps
    cudaStreamCreate(&stream);
    JobShopHeuristic::SolveBatchNew(
        d_problems_, d_evaluators_, d_ops_working, d_results,
        num_problems_to_evaluate_, nn_candidate_count, max_ops_per_problem_,
        stream
    );
    cudaStreamSynchronize(stream);
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "Kernel error during CMA-ES evaluation: " << cudaGetErrorString(kernelErr) << std::endl;
        Eigen::VectorXd bad_result = Eigen::VectorXd::Constant(nn_candidate_count, 1e9);
        cudaFree(d_ops_working);
        cudaFree(d_results);
        cudaStreamDestroy(stream);
        return bad_result;
    }

    auto t6 = std::chrono::high_resolution_clock::now();

    std::vector<float> host_results(nn_candidate_count);
    cudaMemcpy(host_results.data(), d_results, sizeof(float) * nn_candidate_count, cudaMemcpyDeviceToHost);

    auto t7 = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd fvalues(nn_candidate_count);
    for (int r = 0; r < nn_candidate_count; ++r)
        fvalues[r] = static_cast<double>(host_results[r]);

    auto t8 = std::chrono::high_resolution_clock::now();

    cudaFree(d_ops_working);
    cudaFree(d_results);
    cudaStreamDestroy(stream);

    std::cout << "[TIMER][CPU] Weight Update : "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms, "
        << "DeviceEvaluator H2D: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms, "
        << "ops_working memcpy: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << " ms, "
        << "Kernel launch+wait: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count() << " ms, "
        << "Results D2H: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t6).count() << " ms, "
        << "fvalues fill: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t7).count() << " ms, "
        << "Total evaluateCandidates: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t0).count() << " ms"
        << std::endl;

    return fvalues;
}
