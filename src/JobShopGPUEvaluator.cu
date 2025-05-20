#include "JobShopGPUEvaluator.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <chrono>

JobShopGPUEvaluator::JobShopGPUEvaluator(const std::string& problem_file, const std::vector<int>& nn_topology)
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
}

JobShopGPUEvaluator::~JobShopGPUEvaluator() {
    FreeProblemDataGPU();
}

void JobShopGPUEvaluator::FreeProblemDataGPU() {
    if(current_num_problems_ > 0) {
        JobShopDataGPU::FreeBatchGPUData(
            d_problems_, 
            d_jobs_, 
            d_ops_,
            d_eligible_, 
            d_succ_, 
            d_procTimes_,
            current_num_problems_  // Pass the stored count
        );
        current_num_problems_ = 0;
    }
    
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
    current_num_problems_ = batch.size();  // Store the count
    
    int num_problems_on_gpu = 0;
    JobShopDataGPU::UploadBatchToGPU(
        cpu_batch_data_, d_problems_, d_jobs_, d_ops_,
        d_eligible_, d_succ_, d_procTimes_, num_problems_on_gpu
    );
    if (num_problems_on_gpu != current_num_problems_)
        throw std::runtime_error("Mismatch in uploaded problem count");
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

    // Tworzenie DeviceEvaluatorów
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> all_weights; // This seems unused for DeviceEvaluator population, consider removing if not used elsewhere.
    std::vector<float> all_biases;  // This seems unused for DeviceEvaluator population, consider removing if not used elsewhere.
    all_weights.reserve(nn_candidate_count * nn_total_params_ * 2); 
    all_biases.reserve(nn_candidate_count * nn_total_params_);
    
    // Alokujemy wszystko na raz
    std::vector<NeuralNetwork::DeviceEvaluator> host_evaluators(nn_candidate_count);
    std::vector<NeuralNetwork> active_neural_networks; // Store NeuralNetwork objects here
    active_neural_networks.reserve(nn_candidate_count);
    
    // Przetwarzamy kandydatów wsadowo
    for (int r = 0; r < nn_candidate_count; ++r) {
        // Potrzebujemy przechować wagi przed spłaszczeniem
        std::vector<std::vector<float>> weights_for_net; // Renamed to avoid conflict
        std::vector<std::vector<float>> biases_for_net;  // Renamed to avoid conflict
        int paramIdx = 0;
        
        for (size_t i = 1; i < nn_topology_.size(); ++i) {
            int prevLayerSize = nn_topology_[i-1];
            int currLayerSize = nn_topology_[i];
            
            // Prealokujemy wektory
            std::vector<float> layerWeights(prevLayerSize * currLayerSize);
            std::vector<float> layerBiases(currLayerSize);
            
            for (int w = 0; w < prevLayerSize * currLayerSize; ++w)
                layerWeights[w] = static_cast<float>(candidates(paramIdx++, r));
            
            weights_for_net.push_back(layerWeights);
            
            for (int b = 0; b < currLayerSize; ++b)
                layerBiases[b] = static_cast<float>(candidates(paramIdx++, r));
            
            biases_for_net.push_back(layerBiases);
        }
     
        // Create NeuralNetwork and store it in the vector to keep it alive
        active_neural_networks.emplace_back(nn_topology_, &weights_for_net, &biases_for_net);
        host_evaluators[r] = active_neural_networks.back().GetDeviceEvaluator();
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    // Kopiowanie DeviceEvaluatorów na GPU
    NeuralNetwork::DeviceEvaluator* d_evaluators = nullptr;
    cudaMalloc(&d_evaluators, sizeof(NeuralNetwork::DeviceEvaluator) * nn_candidate_count);
    cudaMemcpy(d_evaluators, host_evaluators.data(), sizeof(NeuralNetwork::DeviceEvaluator) * nn_candidate_count, cudaMemcpyHostToDevice);
    auto t3 = std::chrono::high_resolution_clock::now();

    // Przygotowanie ops_working
    std::vector<GPUOperation> ops_working(nn_candidate_count * num_problems_to_evaluate_ * max_ops_per_problem_);
    for (int w = 0; w < nn_candidate_count; ++w) {
      for (int p = 0; p < num_problems_to_evaluate_; ++p) {
        int base = (w * num_problems_to_evaluate_ + p) * max_ops_per_problem_;
        int opsOffset = cpu_batch_data_.operationsOffsets[p];
        int opsCount = cpu_batch_data_.operationsOffsets[p+1] - cpu_batch_data_.operationsOffsets[p];
        // Zawsze kopiuj świeże dane z CPU
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
        d_problems_, d_evaluators, d_ops_working, d_results, 
        num_problems_to_evaluate_, nn_candidate_count, max_ops_per_problem_,
        stream
    );

    cudaStreamSynchronize(stream);

    cudaError_t kernelErr = cudaGetLastError();
    if(kernelErr != cudaSuccess) {
        std::cerr << "Kernel error during CMA-ES evaluation: " << cudaGetErrorString(kernelErr) << std::endl;
        Eigen::VectorXd bad_result = Eigen::VectorXd::Constant(nn_candidate_count, 1e9);
        cudaFree(d_evaluators);
        cudaFree(d_ops_working);
        cudaFree(d_results);
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

    cudaFree(d_evaluators);
    cudaFree(d_ops_working);
    cudaFree(d_results);

    // active_neural_networks will go out of scope here, and their destructors will be called,
    // freeing the GPU memory for weights and biases. This is now safe as the kernel has finished.

    std::cout << "[TIMER][CPU] DeviceEvaluator creation: "
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

// Replace the debug prints at the end of EvaluateCandidates:
    // Sprawdź, czy rozmiar wag i biasów jest poprawnie obliczany:
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