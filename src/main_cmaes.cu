#include "customCMAES.hpp"
#include "JobShopGPUEvaluator.cuh"

/*
:0 will be used to mark OG Stanislaus's code regarding cmaes, 
so that i (or Copilot) dont fuck it up 
*/
using namespace libcmaes;

JobShopGPUEvaluator* g_gpu_test_evaluator = nullptr;
JobShopGPUEvaluator* g_gpu_validate_evaluator = nullptr;

float best_val_makespan = std::numeric_limits<float>::max();
Eigen::VectorXd best_weights; // Will hold the best weights found so far

int main(int argc, char *argv[])
{
    const std::vector<int> topology = {86, 32, 16, 1};
    const int batch_size = 50;
    const std::string test_problem_file = "learn_3k_and_test.json";

    const std::string validate_problem_file = "validate_13k.json"; // copy of learn_1k_and_test.json for now

    //only test data:
    //rnd_JT(5)_J(15)_M(5)_JO(5-10)_O(20)_OM(1-3)_test
    int population_size = 192;//:0

    int nn_weights_count = NeuralNetwork::CalculateTotalParameters(topology);//:0
    
    std::cout<< nn_weights_count <<std::endl;
    
    JobShopGPUEvaluator gpu_evaluator(test_problem_file, topology, population_size, 3100);
    g_gpu_test_evaluator = &gpu_evaluator;

    g_gpu_validate_evaluator = new JobShopGPUEvaluator(validate_problem_file, topology, 1, 350); // 50 problems for validation

    // int total_problems = gpu_evaluator.GetTotalProblems();
    std::vector<double> x0(nn_weights_count, 0.0);
/*     for(int i = 0; i < nn_weights_count; i++) {
        x0[i] = (double)rand() / RAND_MAX * 0.01 - 0.005;
        
    } */
    
    double sigma = 0.1;//:0
    CMAParameters<> cmaparams(x0, sigma, population_size);//:0  
    cmaparams.set_sep();
    cmaparams.set_algo(sepaCMAES);
    
    //const uint64_t fixed_seed = 12345; // Choose any constant value
    //CMAParameters<> cmaparams(x0, sigma, population_size, fixed_seed);

    FitFunc eval = [](const double *x, const int N) -> double { return 0.0; }; //:0
    ESOptimizer<customCMAStrategy,CMAParameters<>> optim(eval, cmaparams);//:0

    int batch_start = 0;
    int global_iter=0;

    while(!optim.stop() && gpu_evaluator.SetCurrentBatch(batch_start, batch_size)) {
        dMat candidates = optim.ask();//:0
        optim.eval(candidates);//:0
        optim.tell();//:0
        optim.inc_iter();//:0

        // best_val_makespan = min(best_val_makespan, optim.get_best_fvalue()); //?
        std::cout << "Best makespan: " << optim.get_best_fvalue() << std::endl;

        batch_start += batch_size;
        global_iter++;
        
        std::cout << "Global iterations: " << global_iter << std::endl;

        if (global_iter % 10 == 0) {
            // 1. Get best weights from optimizer
            auto solutions = optim.get_solutions(); // Returns CMASolutions
            auto best_candidate = solutions.best_candidate(); // Returns Candidate

            // Get the weights from CMA-ES (which are const std::vector<double>&)
            const auto& cma_weights_std_vector = best_candidate.get_x();

            best_weights = Eigen::Map<const Eigen::VectorXd>(cma_weights_std_vector.data(), cma_weights_std_vector.size());

            // The following line correctly creates a copy for current_best_weights
            Eigen::VectorXd current_best_weights(best_weights); 
        
            // 2. Evaluate on validation set (50 problems)
            // Prepare a matrix for a single candidate
            Eigen::MatrixXd best_candidate_matrix(nn_weights_count, 1);
            best_candidate_matrix.col(0) = current_best_weights;
        
            // Set the validation evaluator to the first 50 problems
            g_gpu_validate_evaluator->SetCurrentBatch(batch_start/10, batch_size);
            Eigen::VectorXd val_results = g_gpu_validate_evaluator->EvaluateCandidates(best_candidate_matrix);
        
            float avg_val_makespan = val_results[0]; // Since only one candidate
        
            std::cout << "[VALIDATION] Iter " << global_iter << ": avg makespan = " << avg_val_makespan
                      << " (best so far: " << best_val_makespan << ")" << std::endl;
        
            // 3. If improved, replace population
            if (avg_val_makespan < best_val_makespan) {
                std::cout << "[VALIDATION] New best found! Updating CMA-ES mean." << std::endl;

                best_val_makespan = avg_val_makespan;
                best_weights = current_best_weights;
        
                // Replace all CMA-ES candidates with this best_weights
                // Eigen::MatrixXd new_population(nn_weights_count, population_size); // ? maybe leave this in?
                // for (int i = 0; i < population_size; ++i)                          // ? maybe leave this in?
                //     new_population.col(i) = best_weights;                          // ? maybe leave this in?
                
                auto& solutions = optim.get_solutions();
                solutions.set_xmean(best_weights); // Set new mean
                // solutions.reset(); // Reset with new mean // ?
                    
                // optim.set_solutions(new_population); // You may need to implement this in your optimizer if not present
                // std::cout << "[CMA-ES] Population reset to best validation weights." << std::endl;
            }
        }
        
    }
    //std::cout << optim.get_solutions() << std::endl;//:0
    return 0;
}