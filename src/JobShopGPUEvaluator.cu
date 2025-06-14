#include "JobShopGPUEvaluator.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <chrono>

JobShopGPUEvaluator::JobShopGPUEvaluator(const std::string& problem_file, const std::vector<int>& nn_topology, const int &population_size, const int problem_count)
    : nn_topology_(nn_topology), nn_candidate_count_(population_size)
{
    d_ops_working_ = nullptr;
    current_d_ops_working_size_ = 0;

    // all problems at once 
    cpu_problems_ = JobShopData::LoadFromParallelJson(problem_file, problem_count);//TODO fix nummber of problem assignment 
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

    // Allocate and upload shared topology array
    if (!nn_topology_.empty()) {
        size_t topology_size_bytes = nn_topology_.size() * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_shared_topology_array_, topology_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_shared_topology_array_, nn_topology_.data(), topology_size_bytes, cudaMemcpyHostToDevice));
    } else {
        // Handle empty topology case if necessary, or ensure nn_topology_ is never empty
        d_shared_topology_array_ = nullptr;
    }

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

    // Create a temporary vector for initial DeviceEvaluator setup
    std::vector<NeuralNetwork::DeviceEvaluator> temp_host_evaluators(nn_candidate_count_);

    for (int r = 0; r < nn_candidate_count_; ++r) {
        temp_host_evaluators[r].weights = d_all_candidate_weights_ + (size_t)r * nn_total_weights_per_network_;
        temp_host_evaluators[r].biases  = d_all_candidate_biases_  + (size_t)r * nn_total_biases_per_network_;

        if (nn_topology_.size() > MAX_NN_LAYERS) { // MAX_NN_LAYERS is defined in NeuralNetwork.cuh
            throw std::runtime_error("JobShopGPUEvaluator: Network topology exceeds MAX_NN_LAYERS.");
        }
        for (size_t i = 0; i < nn_topology_.size(); ++i) {
            temp_host_evaluators[r].d_topology[i] = nn_topology_[i];
        }
        
        temp_host_evaluators[r].num_layers = nn_topology_.size();
        temp_host_evaluators[r].max_layer_size = NeuralNetwork::maxLayerSize; // Access static const member
    }

    // Allocate and copy DeviceEvaluators to GPU ONCE
    CUDA_CHECK(cudaMalloc(&d_evaluators_, sizeof(NeuralNetwork::DeviceEvaluator) * nn_candidate_count_));
    CUDA_CHECK(cudaMemcpy(d_evaluators_, temp_host_evaluators.data(), sizeof(NeuralNetwork::DeviceEvaluator) * nn_candidate_count_, cudaMemcpyHostToDevice));
}

JobShopGPUEvaluator::~JobShopGPUEvaluator() {
    FreeProblemDataGPU();
    cudaFree(d_evaluators_);
    cudaFree(d_all_candidate_weights_);
    cudaFree(d_all_candidate_biases_);
    cudaFreeHost(h_pinned_all_weights_);  // Use cudaFreeHost for pinned memory
    cudaFreeHost(h_pinned_all_biases_);  // Use cudaFreeHost for pinned memory
    
    if (d_ops_working_ != nullptr) {
        CUDA_CHECK(cudaFree(d_ops_working_));
        d_ops_working_ = nullptr;
    }

    if (d_shared_topology_array_ != nullptr) {
        CUDA_CHECK(cudaFree(d_shared_topology_array_));
        d_shared_topology_array_ = nullptr;
    }
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
    FreeProblemDataGPU(); // Frees d_problems_, d_jobs_, d_ops_ (reference problem data) etc.
    cpu_batch_data_ = JobShopDataGPU::PrepareBatchCPU(batch);
    num_problems_to_evaluate_ = batch.size();

    int num_problems_on_gpu_check = 0; // Renamed to avoid conflict
    JobShopDataGPU::UploadBatchToGPU(
        cpu_batch_data_, d_problems_, d_jobs_, d_ops_,
        d_eligible_, d_succ_, d_procTimes_, num_problems_on_gpu_check
    );
    if (num_problems_on_gpu_check != num_problems_to_evaluate_)
        throw std::runtime_error("Mismatch in number of problems uploaded to GPU for reference data.");

    // Now handle d_ops_working_
    if (num_problems_to_evaluate_ > 0) { // Only if there are problems to evaluate
        size_t required_total_elements = (size_t)nn_candidate_count_ * num_problems_to_evaluate_ * max_ops_per_problem_;
        size_t required_size_bytes = required_total_elements * sizeof(GPUOperation);

        if (required_size_bytes != current_d_ops_working_size_) {
            if (d_ops_working_ != nullptr) {
                CUDA_CHECK(cudaFree(d_ops_working_));
            }
            CUDA_CHECK(cudaMalloc(&d_ops_working_, required_size_bytes));
            current_d_ops_working_size_ = required_size_bytes;
        }

        // Populate d_ops_working_ by replicating problem data for each NN candidate
        std::vector<GPUOperation> h_ops_working_staging(required_total_elements);
        
        for (int nn_idx = 0; nn_idx < nn_candidate_count_; ++nn_idx) {
            for (int prob_idx_in_batch = 0; prob_idx_in_batch < num_problems_to_evaluate_; ++prob_idx_in_batch) {
                size_t dest_base_elem_offset = (nn_idx * num_problems_to_evaluate_ + prob_idx_in_batch) * max_ops_per_problem_;
                
                int src_ops_offset_in_batch_buffer = cpu_batch_data_.operationsOffsets[prob_idx_in_batch];
                int src_ops_count = cpu_batch_data_.operationsOffsets[prob_idx_in_batch + 1] - src_ops_offset_in_batch_buffer;

                if (src_ops_count > max_ops_per_problem_) {
                    // This is an issue: problem has more ops than allocated space per problem.
                    // Consider throwing an error or logging. For now, truncate (dangerous).
                    fprintf(stderr, "Warning: Problem %d has %d ops, exceeding max_ops_per_problem_ %d. Truncating.\n",
                            prob_idx_in_batch, src_ops_count, max_ops_per_problem_);
                    src_ops_count = max_ops_per_problem_;
                }
                
                if (src_ops_count > 0) {
                     memcpy(&h_ops_working_staging[dest_base_elem_offset], 
                            &cpu_batch_data_.operations[src_ops_offset_in_batch_buffer], 
                            src_ops_count * sizeof(GPUOperation));
                }

                // Zero out the remaining part of the slot for this problem if necessary
                if (src_ops_count < max_ops_per_problem_) {
                    memset(&h_ops_working_staging[dest_base_elem_offset + src_ops_count], 
                           0, 
                           (max_ops_per_problem_ - src_ops_count) * sizeof(GPUOperation));
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(d_ops_working_, h_ops_working_staging.data(), current_d_ops_working_size_, cudaMemcpyHostToDevice));
    } else { // No problems to evaluate, free d_ops_working_ if it exists
        if (d_ops_working_ != nullptr) {
            CUDA_CHECK(cudaFree(d_ops_working_));
            d_ops_working_ = nullptr;
            current_d_ops_working_size_ = 0;
        }
    }
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

__global__ void UpdateEvaluatorPointersKernel(
    NeuralNetwork::DeviceEvaluator* d_evaluators,
    float* d_all_weights,
    float* d_all_biases,
    int nn_total_weights_per_network,
    int nn_total_biases_per_network, // Add this parameter
    int nn_candidate_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nn_candidate_count) {
        d_evaluators[idx].weights = d_all_weights + (size_t)idx * nn_total_weights_per_network;
        d_evaluators[idx].biases  = d_all_biases  + (size_t)idx * nn_total_biases_per_network; // Corrected
    }
}


Eigen::VectorXd JobShopGPUEvaluator::EvaluateCandidates(const Eigen::MatrixXd& candidates, const bool& validation_mode) {
    auto t0 = std::chrono::high_resolution_clock::now();

    int nn_candidate_count = candidates.cols();
    if (candidates.rows() != nn_total_params_)
        throw std::runtime_error("Mismatch in number of weights per NN candidate.");

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

    // Launch the pointer update kernel
    int threadsPerBlock = 128; // Adjust as needed
    int blocks = (nn_candidate_count + threadsPerBlock - 1) / threadsPerBlock;
    UpdateEvaluatorPointersKernel<<<blocks, threadsPerBlock, 0, stream>>>(
        d_evaluators_,
        d_all_candidate_weights_,
        d_all_candidate_biases_,
        nn_total_weights_per_network_, // Already a class member
        nn_total_biases_per_network_,  // Already a class member
        nn_candidate_count
    );

    auto t3 = std::chrono::high_resolution_clock::now();

    auto t4 = std::chrono::high_resolution_clock::now();

    float* d_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(float) * nn_candidate_count));

    // Kernel
    auto t5 = std::chrono::high_resolution_clock::now();

    JobShopHeuristic::SolveBatchNew(
        d_problems_, d_evaluators_, d_ops_working_, d_results, // USE THE CLASS MEMBER d_ops_working_
        num_problems_to_evaluate_, // This is numProblems_per_block for the kernel
        nn_candidate_count_, // This is numWeights_total_blocks for the kernel
        nn_total_weights_per_network_, // <<< Pass the total weights for one NN here
        nn_total_biases_per_network_, // <<< Pass the total biases for one NN here
        max_ops_per_problem_,
        stream, // Pass the stream
        nn_total_params_, // <<< Pass the total parameters for one NN here
        validation_mode
    );

    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto t6 = std::chrono::high_resolution_clock::now();

    std::vector<float> host_results(nn_candidate_count);
    CUDA_CHECK(cudaMemcpy(host_results.data(), d_results, sizeof(float) * nn_candidate_count, cudaMemcpyDeviceToHost));

    auto t7 = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd fvalues(nn_candidate_count);
    for (int r = 0; r < nn_candidate_count; ++r)
        fvalues[r] = static_cast<double>(host_results[r]);

    auto t8 = std::chrono::high_resolution_clock::now();

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

float JobShopGPUEvaluator::EvaluateForMinMakespan(const Eigen::VectorXd& candidate_weights, int num_problems) {
    // 1. Set the evaluator to use the specified number of validation problems.
    if (!SetCurrentBatch(0, num_problems)) {
        std::cerr << "[ERROR] Could not set batch for validation." << std::endl;
        return std::numeric_limits<float>::max();
    }

    // 2. Prepare a weight matrix with 192 IDENTICAL columns.
    // This "tricks" the evaluator into using all 192 blocks for one network.
    Eigen::MatrixXd replicated_candidate_matrix(nn_total_params_, nn_candidate_count_);
    for (int i = 0; i < nn_candidate_count_; ++i) {
        replicated_candidate_matrix.col(i) = candidate_weights;
    }

    // 3. Evaluate all 192 (identical) candidates on the GPU.
    // The existing EvaluateCandidates function will now correctly distribute
    // the 'num_problems' across all 'nn_candidate_count_' (192) blocks.
    Eigen::VectorXd result_vector = EvaluateCandidates(replicated_candidate_matrix, true);

    // 4. Find the minimum makespan from the results of all blocks.
    // Each element of result_vector is the min makespan from one block's subset of problems.
    // We need the overall minimum.
    if (result_vector.size() > 0) {
        return result_vector.minCoeff();
    }

    return std::numeric_limits<float>::max(); // Return max float on error
}
