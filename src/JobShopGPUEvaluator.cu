#include "JobShopGPUEvaluator.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <chrono>

JobShopGPUEvaluator::JobShopGPUEvaluator(const std::string& problem_file, const std::vector<int>& nn_topology)
    : nn_topology_(nn_topology)
{
    // all problems at once 
    cpu_problems_ = JobShopData::LoadFromParallelJson(problem_file, 500);//TODO fix nummber of problem assignment 
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
}

JobShopGPUEvaluator::~JobShopGPUEvaluator() {
    FreeProblemDataGPU();
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
    auto t0_total = std::chrono::high_resolution_clock::now();
    int nn_candidate_count = candidates.cols();
    if (candidates.rows() != nn_total_params_)
        throw std::runtime_error("Mismatch in number of weights per NN candidate.");

    auto t1_prep = std::chrono::high_resolution_clock::now();

    size_t single_nn_weights_count = 0;
    size_t single_nn_biases_count = 0;
    for (size_t i = 1; i < nn_topology_.size(); ++i) {
        single_nn_weights_count += nn_topology_[i-1] * nn_topology_[i];
        single_nn_biases_count += nn_topology_[i];
    }

    size_t total_weights_all_candidates_size = single_nn_weights_count * nn_candidate_count;
    size_t total_biases_all_candidates_size = single_nn_biases_count * nn_candidate_count;

    float* d_all_candidate_weights_mega_buffer = nullptr;
    float* d_all_candidate_biases_mega_buffer = nullptr;
    cudaMalloc(&d_all_candidate_weights_mega_buffer, total_weights_all_candidates_size * sizeof(float));
    cudaMalloc(&d_all_candidate_biases_mega_buffer, total_biases_all_candidates_size * sizeof(float));

    std::vector<NeuralNetwork::DeviceEvaluator> host_evaluators(nn_candidate_count);
    
    std::vector<float> temp_host_weights_buffer(single_nn_weights_count);
    std::vector<float> temp_host_biases_buffer(single_nn_biases_count);

    size_t current_mega_weight_offset_elements = 0;
    size_t current_mega_bias_offset_elements = 0;
    
    if (nn_topology_.size() > MAX_NN_LAYERS) { // MAX_NN_LAYERS should be accessible here or use a constant
        throw std::runtime_error("NN topology size exceeds MAX_NN_LAYERS");
    }

    for (int r = 0; r < nn_candidate_count; ++r) {
        int paramIdx = 0;
        size_t current_temp_weight_idx = 0;
        size_t current_temp_bias_idx = 0;
        
        for (size_t i = 1; i < nn_topology_.size(); ++i) {
            int prevLayerSize = nn_topology_[i-1];
            int currLayerSize = nn_topology_[i];
            
            for (int w = 0; w < prevLayerSize * currLayerSize; ++w)
                temp_host_weights_buffer[current_temp_weight_idx++] = static_cast<float>(candidates(paramIdx++, r));
            
            for (int b = 0; b < currLayerSize; ++b)
                temp_host_biases_buffer[current_temp_bias_idx++] = static_cast<float>(candidates(paramIdx++, r));
        }

        cudaMemcpy(d_all_candidate_weights_mega_buffer + current_mega_weight_offset_elements,
                   temp_host_weights_buffer.data(),
                   single_nn_weights_count * sizeof(float),
                   cudaMemcpyHostToDevice);
        
        cudaMemcpy(d_all_candidate_biases_mega_buffer + current_mega_bias_offset_elements,
                   temp_host_biases_buffer.data(),
                   single_nn_biases_count * sizeof(float),
                   cudaMemcpyHostToDevice);

        host_evaluators[r].d_weights = d_all_candidate_weights_mega_buffer + current_mega_weight_offset_elements;
        host_evaluators[r].d_biases = d_all_candidate_biases_mega_buffer + current_mega_bias_offset_elements;
        
        // Copy topology to embedded array
        memcpy(host_evaluators[r].d_topology, nn_topology_.data(), nn_topology_.size() * sizeof(int));
        for (size_t k_topo = nn_topology_.size(); k_topo < MAX_NN_LAYERS; ++k_topo) { // MAX_NN_LAYERS needs to be visible
            host_evaluators[r].d_topology[k_topo] = 0; // Pad with zeros
        }
        host_evaluators[r].num_layers = nn_topology_.size();

        current_mega_weight_offset_elements += single_nn_weights_count;
        current_mega_bias_offset_elements += single_nn_biases_count;
    }
    auto t2_eval_prep = std::chrono::high_resolution_clock::now();

    NeuralNetwork::DeviceEvaluator* d_evaluators = nullptr;
    cudaMalloc(&d_evaluators, sizeof(NeuralNetwork::DeviceEvaluator) * nn_candidate_count);
    cudaMemcpy(d_evaluators, host_evaluators.data(), sizeof(NeuralNetwork::DeviceEvaluator) * nn_candidate_count, cudaMemcpyHostToDevice);
    auto t3_eval_h2d = std::chrono::high_resolution_clock::now();

    // GPUOperation* d_ops_working = nullptr; // Deklaracja już istnieje jako pole klasy, ale tu chodzi o lokalną zmienną w funkcji
                                          // Jednakże, d_ops_working w tym kontekście jest buforem roboczym dla kernela.
                                          // d_ops_ (pole klasy) to szablon danych problemu.

    // Alokuj pamięć dla d_ops_working, ale nie kopiuj z hosta
    // Ten bufor będzie używany przez kernel jako pamięć robocza
    GPUOperation* d_kernel_ops_working_buffer = nullptr; // Użyj innej nazwy, aby uniknąć konfliktu z polem klasy d_ops_
    size_t d_kernel_ops_working_size_bytes = (size_t)nn_candidate_count * num_problems_to_evaluate_ * max_ops_per_problem_ * sizeof(GPUOperation);
    if (d_kernel_ops_working_size_bytes == 0) {
        throw std::runtime_error("Calculated size for d_kernel_ops_working_buffer is zero.");
    }
    cudaError_t err_alloc_ops_working = cudaMalloc(&d_kernel_ops_working_buffer, d_kernel_ops_working_size_bytes);
    if (err_alloc_ops_working != cudaSuccess) {
        throw std::runtime_error("Failed to allocate d_kernel_ops_working_buffer: " + std::string(cudaGetErrorString(err_alloc_ops_working)));
    }
    
    // Usuwamy tworzenie std::vector<GPUOperation> ops_working na CPU i jego kopiowanie
    // auto t4_ops_memcpy_cpu = std::chrono::high_resolution_clock::now(); // Timer niepotrzebny
    // auto t5_ops_h2d = std::chrono::high_resolution_clock::now(); // Timer niepotrzebny

    float* d_results = nullptr;
    cudaMalloc(&d_results, sizeof(float) * nn_candidate_count);

    auto t6_kernel_launch = std::chrono::high_resolution_clock::now();
    cudaStream_t stream; 
    cudaStreamCreate(&stream);

     JobShopHeuristic::SolveBatchNew(
        d_problems_,          // Wskaźnik do struktur GPUProblem (zawierają wskaźniki do szablonów danych)
        d_evaluators,         // Wskaźnik do ewaluatorów sieci neuronowych
        d_kernel_ops_working_buffer, // Pamięć robocza dla kernela do wypełnienia i modyfikacji
        d_results,            // Bufor na wyniki
        num_problems_to_evaluate_, 
        nn_candidate_count, 
        max_ops_per_problem_, 
        stream
    );

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream); 
    auto t7_kernel_sync = std::chrono::high_resolution_clock::now();

    std::vector<float> host_results(nn_candidate_count);
    cudaMemcpy(host_results.data(), d_results, sizeof(float) * nn_candidate_count, cudaMemcpyDeviceToHost);
    auto t8_results_d2h = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd fvalues(nn_candidate_count);
    for (int r = 0; r < nn_candidate_count; ++r)
        fvalues[r] = static_cast<double>(host_results[r]);
    
    auto t9_fvalues_fill = std::chrono::high_resolution_clock::now();
    double min_makespan = (fvalues.size() > 0) ? fvalues.minCoeff() : 0.0;
    std::cout << "[INFO] Best average makespan: " << min_makespan << std::endl;

    cudaFree(d_all_candidate_weights_mega_buffer);
    cudaFree(d_all_candidate_biases_mega_buffer);
    cudaFree(d_evaluators);
    cudaFree(d_kernel_ops_working_buffer); // Zwolnij pamięć roboczą
    cudaFree(d_results);
   
    // Update timers
    std::cout << "[TIMER][CPU] Evaluator NN Data Prep: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2_eval_prep - t1_prep).count() << " ms, "
              << "Evaluator Struct H2D: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3_eval_h2d - t2_eval_prep).count() << " ms, "
              << "Kernel launch+sync: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t7_kernel_sync - t6_kernel_launch).count() << " ms, "
              << "Results D2H: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t8_results_d2h - t7_kernel_sync).count() << " ms, "
              << "fvalues fill: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t9_fvalues_fill - t8_results_d2h).count() << " ms, "
              << "Total evaluateCandidates: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t9_fvalues_fill - t0_total).count() << " ms"
              << std::endl;

// Check the sizes
    int total_weights = 0;
    int total_biases = 0;
    for(int i = 1; i < nn_topology_.size(); i++) {
        total_weights += nn_topology_[i-1] * nn_topology_[i];
        total_biases += nn_topology_[i];
    }

    std::cout << "[DEBUG] Total weights calculated: " << total_weights << std::endl;
    std::cout << "[DEBUG] Total biases calculated: " << total_biases << std::endl;
    std::cout << "[DEBUG] Total parameters: " << nn_total_params_ << std::endl;

    return fvalues;
}