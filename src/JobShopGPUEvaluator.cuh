#ifndef JOB_SHOP_GPU_EVALUATOR_CUH
#define JOB_SHOP_GPU_EVALUATOR_CUH

#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>
#include "JobShopData.cuh"
#include "NeuralNetwork.cuh"
#include "JobShopHeuristic.cuh"

class JobShopGPUEvaluator {
public:
    JobShopGPUEvaluator(const std::string& problem_file, const std::vector<int>& nn_topology, const int &population_size, const int problem_count, int problem_offset, int max_loaded_problems);
    ~JobShopGPUEvaluator();

    // Ustawia batch problemów do ewaluacji (nie kopiuje z pliku, tylko z RAM)
    bool SetCurrentBatch(int batch_start, int batch_size);

    // Ewaluacja populacji wag na aktualnym batchu problemów
    Eigen::VectorXd EvaluateCandidates(const Eigen::MatrixXd& candidates, const bool& validation_mode);
    
    float EvaluateForMinMakespan(const Eigen::VectorXd& candidate_weights, int num_problems);

    int GetTotalNNParams() const { return nn_total_params_; }
    int GetTotalProblems() const { return cpu_problems_.size(); }

private:
    void FreeProblemDataGPU();
    void PrepareProblemDataGPU(const std::vector<JobShopData>& batch);

    std::vector<JobShopData> cpu_problems_; // wszystkie problemy w RAM
    int num_problems_to_evaluate_ = 0;
    std::vector<int> nn_topology_;
    int nn_total_params_ = 0;
    int max_ops_per_problem_ = 0;

    // GPU pointers
    GPUProblem* d_problems_ = nullptr;
    GPUJob* d_jobs_ = nullptr;
    GPUOperation* d_ops_ = nullptr; 
    int* d_eligible_ = nullptr;
    int* d_succ_ = nullptr;
    int* d_procTimes_ = nullptr;

    BatchJobShopGPUData cpu_batch_data_; // aktualny batch

    int nn_candidate_count_ = 0; // number of candidates

    int* d_shared_topology_array_ = nullptr;
    
    NeuralNetwork::DeviceEvaluator* d_evaluators_ = nullptr; // Evaluators on the device
    GPUOperation* d_ops_working_ = nullptr;      // Writable operations buffer for all (NN, problem) pairs
    size_t current_d_ops_working_size_ = 0; // To track current allocated size


    float* d_all_candidate_weights_ = nullptr;
    float* d_all_candidate_biases_ = nullptr;
    float* h_pinned_all_weights_ = nullptr;
    float* h_pinned_all_biases_ = nullptr;
    size_t total_weights_size_ = 0;
    size_t total_biases_size_ = 0;
    int nn_total_weights_per_network_ = 0; // Store total weights per network
    int nn_total_biases_per_network_ = 0; // Store total biases per network
};

__global__ void UpdateEvaluatorPointersKernel(
    NeuralNetwork::DeviceEvaluator* d_evaluators,
    float* d_all_weights,
    float* d_all_biases,
    int nn_total_weights_per_network, // Stride for weights array
    int nn_total_biases_per_network,  // Stride for biases array
    int nn_candidate_count);


#endif // JOB_SHOP_GPU_EVALUATOR_CUH