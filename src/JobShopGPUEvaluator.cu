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

    nn_total_weights_per_network_ = 0;
    nn_total_biases_per_network_ = 0;
    for (size_t i = 1; i < nn_topology_.size(); ++i) {
        nn_total_weights_per_network_ += nn_topology_[i - 1] * nn_topology_[i];
        nn_total_biases_per_network_ += nn_topology_[i];
    }

    total_weights_size_ = (size_t)nn_candidate_count_ * nn_total_weights_per_network_ * sizeof(float);
    total_biases_size_ = (size_t)nn_candidate_count_ * nn_total_biases_per_network_ * sizeof(float);

    // Allocate pinned host memory
    CUDA_CHECK(cudaHostAlloc(&h_pinned_all_weights_, total_weights_size_, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_pinned_all_biases_, total_biases_size_, cudaHostAllocDefault));

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_all_candidate_weights_, total_weights_size_));
    CUDA_CHECK(cudaMalloc(&d_all_candidate_biases_, total_biases_size_));

    for (int r = 0; r < nn_candidate_count_; ++r) {
        neural_networks_[r] = NeuralNetwork(nn_topology_); // Create a NeuralNetwork
        neural_networks_[r].cudaData->d_weights = d_all_candidate_weights_ + r * nn_total_weights_per_network_;
        neural_networks_[r].cudaData->d_biases = d_all_candidate_biases_ + r * nn_total_biases_per_network_;
        host_evaluators_[r] = neural_networks_[r].GetDeviceEvaluator(); // Get its DeviceEvaluator
    }    

    // Allocate and copy DeviceEvaluators to GPU
    cudaMalloc(&d_evaluators_, sizeof(NeuralNetwork::DeviceEvaluator) * nn_candidate_count_);
    cudaMemcpy(d_evaluators_, host_evaluators_.data(), sizeof(NeuralNetwork::DeviceEvaluator) * nn_candidate_count_, cudaMemcpyHostToDevice);
}

JobShopGPUEvaluator::~JobShopGPUEvaluator() {
    FreeProblemDataGPU();
    cudaFree(d_evaluators_);
    cudaFree(d_all_candidate_weights_);
    cudaFree(d_all_candidate_biases_);
    cudaFreeHost(h_pinned_all_weights_);  // Use cudaFreeHost for pinned memory
    cudaFreeHost(h_pinned_all_biases_);  // Use cudaFreeHost for pinned memory
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

    // --------------------------------------------------------------------
    // Consolidated Weight/Bias Update and Asynchronous Transfer
    // --------------------------------------------------------------------

    auto t1 = std::chrono::high_resolution_clock::now();

    // 1. Populate Pinned Host Memory: Directly copy from Eigen matrix to the pinned host buffers
    for (int r = 0; r < nn_candidate_count; ++r) {
        int paramIdx = 0;
        size_t weight_offset = (size_t)r * nn_total_weights_per_network_;
        size_t bias_offset = (size_t)r * nn_total_biases_per_network_;

        for (size_t i = 1; i < nn_topology_.size(); ++i) {
            int prevLayerSize = nn_topology_[i - 1];
            int currLayerSize = nn_topology_[i];

            // Weights
            for (int w = 0; w < prevLayerSize * currLayerSize; ++w) {
                h_pinned_all_weights_[weight_offset++] = static_cast<float>(candidates(paramIdx++, r));
            }

            // Biases
            for (int b = 0; b < currLayerSize; ++b) {
                h_pinned_all_biases_[bias_offset++] = static_cast<float>(candidates(paramIdx++, r));
            }
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    // 2. Asynchronous Memory Transfer: Copy all weights and biases in single calls
    cudaStream_t stream; // Get the stream (assuming this is a member variable now)
    cudaStreamCreate(&stream);
    CUDA_CHECK(cudaMemcpyAsync(d_all_candidate_weights_, h_pinned_all_weights_, total_weights_size_, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_all_candidate_biases_, h_pinned_all_biases_, total_biases_size_, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Update DeviceEvaluators on the host with the new weight/bias pointers
    for (int r = 0; r < nn_candidate_count; ++r) {
        neural_networks_[r].cudaData->d_weights = d_all_candidate_weights_ + r * nn_total_weights_per_network_;
        neural_networks_[r].cudaData->d_biases = d_all_candidate_biases_ + r * nn_total_biases_per_network_;
        host_evaluators_[r] = neural_networks_[r].GetDeviceEvaluator(); // Get its DeviceEvaluator
    }
    // Copy the updated DeviceEvaluators to the GPU
    CUDA_CHECK(cudaMemcpyAsync(d_evaluators_, host_evaluators_.data(), sizeof(NeuralNetwork::DeviceEvaluator) * nn_candidate_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // --------------------------------------------------------------------
    // Rest of EvaluateCandidates (Kernel Launch, Result Collection)
    // --------------------------------------------------------------------

    auto t3 = std::chrono::high_resolution_clock::now();

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
    CUDA_CHECK(cudaMalloc(&d_ops_working, ops_working.size() * sizeof(GPUOperation)));
    CUDA_CHECK(cudaMemcpy(d_ops_working, ops_working.data(), ops_working.size() * sizeof(GPUOperation), cudaMemcpyHostToDevice));

    float* d_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(float) * nn_candidate_count));

    // Kernel
    auto t5 = std::chrono::high_resolution_clock::now();

    JobShopHeuristic::SolveBatchNew(
        d_problems_, d_evaluators_, d_ops_working, d_results,
        num_problems_to_evaluate_, // This is numProblems_per_block for the kernel
        nn_candidate_count_,       // This is numWeights_total_blocks for the kernel
        max_ops_per_problem_,
        stream,                    // Pass the stream
        nn_total_params_           // <<< Pass the total parameters for one NN here
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));

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
    CUDA_CHECK(cudaMemcpy(host_results.data(), d_results, sizeof(float) * nn_candidate_count, cudaMemcpyDeviceToHost));

    auto t7 = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd fvalues(nn_candidate_count);
    for (int r = 0; r < nn_candidate_count; ++r)
        fvalues[r] = static_cast<double>(host_results[r]);

    auto t8 = std::chrono::high_resolution_clock::now();

    double min_makespan = (fvalues.size() > 0) ? fvalues.minCoeff() : 0.0;
    std::cout << "[INFO] Best average makespan: " << min_makespan << std::endl;

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


