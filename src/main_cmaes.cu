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

    const std::string test_run_name = "OLD/learn_3k_and_test";
    const std::string test_problem_file = test_run_name + ".json";

    const std::string validate_problem_file = "OLD/validate_13k.json"; // copy of learn_1k_and_test.json for now

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
                best_val_makespan = avg_val_makespan;
                best_weights = current_best_weights;
                
                try {
                    // 1. Construct a filesystem path object.
                    const std::filesystem::path weights_path(test_run_name + "_best_weights.csv");

                    // 2. Extract the parent directory from the full path.
                    // For "OLD/file.csv", this will be "OLD".
                    std::filesystem::path dir_path = weights_path.parent_path();

                    // 3. Create the directory if it's not empty and doesn't exist.
                    // create_directories handles nested paths and doesn't fail if the dir already exists.
                    if (!dir_path.empty()) {
                        std::filesystem::create_directories(dir_path);
                    }

                    // 4. Open the file for writing. This should now succeed.
                    std::cout << "[IO] Saving new best weights to: " << weights_path << std::endl;
                    std::ofstream file(weights_path);
                    if (file.is_open()) {
                        const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
                        file << best_weights.transpose().format(CSVFormat);
                        file.close();
                    } else {
                        // This error is now less likely, but kept for robustness.
                        std::cerr << "[ERROR] Unable to open file for writing: " << weights_path << std::endl;
                    }
                } catch (const std::filesystem::filesystem_error& e) {
                    // Catch potential filesystem errors (e.g., permissions).
                    std::cerr << "[ERROR] Filesystem error: " << e.what() << std::endl;
                }
            }
        }
        
    }
    //std::cout << optim.get_solutions() << std::endl;//:0
    return 0;
}