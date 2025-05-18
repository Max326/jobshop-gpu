#include "JobShopGPUEvaluator.cuh"
#include "NeuralNetwork.cuh"
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <algorithm>

JobShopGPUEvaluator* g_gpu_evaluator = nullptr;

Eigen::MatrixXd GenerateTestWeights(int params, int candidates, float min=-5.0f, float max=5.0f) {
    Eigen::MatrixXd weights(params, candidates);
    std::default_random_engine generator(42); // Fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0, 0.5);
    
    for(int c = 0; c < candidates; ++c) {
        for(int p = 0; p < params; ++p) {
            float val = dist(generator);
            // Clamp values to prevent numerical instability
            weights(p, c) = std::max(min, std::min(max, val));
        }
    }
    return weights;
}

int main(int argc, char *argv[]) {
    const std::vector<int> topology = {86, 32, 16, 1};
    const int batch_size = 1;      // Problems per evaluation
    const std::string problem_file = "test_10k.json";
    const int population_size = 1;  // Number of weight sets to test

    try {
        // 1. Initialize gpu_evaluator
        JobShopGPUEvaluator gpu_evaluator(problem_file, topology);
        g_gpu_evaluator = &gpu_evaluator;
        
        // 2. Generate test weights directly
        const int nn_weights_count = NeuralNetwork::CalculateTotalParameters(topology);
        Eigen::MatrixXd candidates = GenerateTestWeights(nn_weights_count, population_size);
        
        std::cout << "Generated " << population_size << " weight sets with "
                  << nn_weights_count << " parameters each\n";

        // 3. Evaluate single batch
        if(gpu_evaluator.SetCurrentBatch(0, batch_size)) {
            Eigen::VectorXd results = gpu_evaluator.EvaluateCandidates(candidates);
            
            std::cout << "\nEvaluation Results:\n";
            for(int i = 0; i < results.size(); ++i) {
                std::cout << "Weight Set " << i << " | Avg Makespan: " 
                          << results[i] << "\n";
            }
        }
        else {
            std::cerr << "Failed to set evaluation batch\n";
            return 1;
        }

    
    }
    catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // double sigma = 0.1;//:0
    // CMAParameters<> cmaparams(x0, sigma, population_size);//:0  
    
    // //const uint64_t fixed_seed = 12345; // Choose any constant value
    // //CMAParameters<> cmaparams(x0, sigma, population_size, fixed_seed);

    // FitFunc eval = [](const double *x, const int N) -> double { return 0.0; }; //:0
    // ESOptimizer<customCMAStrategy,CMAParameters<>> optim(eval, cmaparams);//:0

    // int batch_start = 0;

    // while(!optim.stop() && gpu_evaluator.SetCurrentBatch(batch_start, batch_size)) {
    //     dMat candidates = optim.ask();//:0
    //     optim.eval(candidates);//:0
    //     optim.tell();//:0
    //     optim.inc_iter();//:0
    //     batch_start += batch_size;
    // }
    return 0;
}
