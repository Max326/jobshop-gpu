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
    JobShopGPUEvaluator(const std::string& problem_file, const std::vector<int>& nn_topology);
    ~JobShopGPUEvaluator();

    // Ustawia batch problemów do ewaluacji (nie kopiuje z pliku, tylko z RAM)
    bool SetCurrentBatch(int batch_start, int batch_size);

    // Ewaluacja populacji wag na aktualnym batchu problemów
    Eigen::VectorXd EvaluateCandidates(const Eigen::MatrixXd& candidates);

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
};

#endif // JOB_SHOP_GPU_EVALUATOR_CUH