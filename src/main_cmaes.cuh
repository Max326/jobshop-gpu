#include "customCMAES.hpp"
#include "JobShopGPUEvaluator.cuh"
#include <fstream>
#include <filesystem>
#include <regex>

/*
:0 will be used to mark OG Stanislaus's code regarding cmaes, 
so that i (or Copilot) dont fuck it up 
*/
using namespace libcmaes;

JobShopGPUEvaluator* g_gpu_train_evaluator = nullptr;
JobShopGPUEvaluator* g_gpu_validate_evaluator = nullptr;
JobShopGPUEvaluator* g_gpu_test_evaluator = nullptr;

float best_val_makespan = std::numeric_limits<float>::max();
Eigen::VectorXd best_weights; 

int main_cmaes(const std::string problem_file, const int max_loaded_problems)
{
    // --- CONFIG ---
    const std::vector<int> topology = {81, 32, 16, 1};
    const int batch_size = 50;
    const int train_problem_count = 130000; //130k --> 2600 iterations
    const int validation_problem_count = 10000; 
    const int test_problem_count = 100;

    const int train_population_size = 192;
    const int valid_and_test_population_size = 1;
    
    int nn_weights_count = NeuralNetwork::CalculateTotalParameters(topology);
    std::vector<double> x0(nn_weights_count, 0.0);
    double sigma = 0.1;
    CMAParameters<> cmaparams(x0, sigma, train_population_size);  
    cmaparams.set_sep();
    cmaparams.set_algo(sepaCMAES);

    FitFunc eval = [](const double *x, const int N) -> double { return 0.0; };
    ESOptimizer<customCMAStrategy,CMAParameters<>> optim(eval, cmaparams);

    const std::string train_problem_file = "TRAIN/" + problem_file + "_total.json";
    const std::string validate_problem_file = "VALID/" + problem_file + "_validation.json";
    const std::string test_problem_file = "TEST/" + problem_file + "_test.json";

    const std::string dataset_name = problem_file;
    std::filesystem::path results_dir = std::filesystem::path("data") / "RESULTS" / dataset_name;
    std::filesystem::create_directories(results_dir);

    std::ofstream train_makespan_file((results_dir / "best_train_makespans.csv").string());
    std::ofstream val_makespan_file((results_dir / "best_val_makespans.csv").string());
    train_makespan_file << "iteration,best_train_avg_makespan\n";
    val_makespan_file << "iteration,val_avg_makespan\n";

    int global_iter = 0;
    int problems_processed = 0;

    if (g_gpu_validate_evaluator) {
        delete g_gpu_validate_evaluator;
    }
    g_gpu_validate_evaluator = new JobShopGPUEvaluator(validate_problem_file, topology, valid_and_test_population_size, validation_problem_count);

    while (problems_processed < train_problem_count) {
        int to_load = std::min(max_loaded_problems, train_problem_count - problems_processed);

        // TODO memory allocation outside of the loop
        JobShopGPUEvaluator gpu_evaluator(train_problem_file, topology, train_population_size, train_problem_count, problems_processed, to_load);
        g_gpu_train_evaluator = &gpu_evaluator;

        int batch_start = 0;
        while (batch_start < to_load && gpu_evaluator.SetCurrentBatch(batch_start, batch_size)) {
            dMat candidates = optim.ask();//:0
            optim.eval(candidates);//:0
            optim.tell();//:0
            optim.inc_iter();//:0

            float best_train_makespan = optim.get_best_fvalue();

            train_makespan_file << global_iter << "," << best_train_makespan << "\n";
            train_makespan_file.flush();

            batch_start += batch_size;
            global_iter++;

            std::cout << "Global iterations: " << global_iter << ", best train makespan: " << best_train_makespan << std::endl;

            // --- VALIDATE  ---
            if (global_iter % 10 == 0) {
                const auto& cma_weights = optim.get_solutions().best_candidate().get_x();
                Eigen::VectorXd current_best_weights = Eigen::Map<const Eigen::VectorXd>(cma_weights.data(), cma_weights.size());

                std::cout << "[VALIDATION] Iter " << global_iter 
                        << ": Running validation on " << validation_problem_count << " problems..." << std::endl;

                float avg_val_makespan = g_gpu_validate_evaluator->EvaluateForMinMakespan(current_best_weights, validation_problem_count);

                std::cout << "[VALIDATION] Average makespan = " << avg_val_makespan
                        << " (best so far: " << best_val_makespan << ")" << std::endl;

                val_makespan_file << global_iter << "," << avg_val_makespan << "\n";
                val_makespan_file.flush();

                if (avg_val_makespan < best_val_makespan) {
                    std::cout << "[VALIDATION] New best network found!" << std::endl;
                    best_val_makespan = avg_val_makespan;
                    best_weights = current_best_weights;

                    try {
                        const std::filesystem::path weights_path("best_weights.csv");
                        std::filesystem::path dir_path = weights_path.parent_path();
                        if (!dir_path.empty()) {
                            std::filesystem::create_directories(dir_path);
                        }
                        std::cout << "[IO] Saving new best weights to: " << weights_path << std::endl;
                        std::ofstream file(weights_path);
                        if (file.is_open()) {
                            const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
                            file << best_weights.transpose().format(CSVFormat);
                            file.close();
                        } else {
                            std::cerr << "[ERROR] Unable to open file for writing: " << weights_path << std::endl;
                        }
                    } catch (const std::filesystem::filesystem_error& e) {
                        std::cerr << "[ERROR] Filesystem error: " << e.what() << std::endl;
                    }
                }
            }
        }
        problems_processed += to_load;
    }

    train_makespan_file.close();
    val_makespan_file.close();

    // --- TEST  ---
    try {
        // JobShopGPUEvaluator test_evaluator(test_problem_file, topology, train_population_size, test_problem_count);
        g_gpu_test_evaluator = new JobShopGPUEvaluator(test_problem_file, topology, valid_and_test_population_size, test_problem_count);

        std::cout << "[TEST] Evaluating best weights on test set (" << test_problem_count << " problems)..." << std::endl;

        float test_avg_makespan = g_gpu_test_evaluator->EvaluateForMinMakespan(best_weights, test_problem_count);

        // save the test result
        std::ofstream test_result_file((results_dir / "best_test_result.csv").string());
        if (test_result_file.is_open()) {
            test_result_file << "avg_makespan," << test_avg_makespan << "\n";
            test_result_file << "weights,";
            for (int i = 0; i < best_weights.size(); ++i) {
                test_result_file << best_weights[i];
                if (i < best_weights.size() - 1) test_result_file << ",";
            }
            test_result_file << "\n";
            test_result_file.close();
            std::cout << "[TEST] Test result saved to best_test_result.csv" << std::endl;
        } else {
            std::cerr << "[ERROR] Unable to open best_test_result.csv for writing!" << std::endl;
        }

        /*         // Save the makespan for each task in the test set
        Eigen::MatrixXd single_candidate_matrix(best_weights.size(), 1);
        single_candidate_matrix.col(0) = best_weights;

        Eigen::VectorXd test_makespans = g_gpu_test_evaluator->EvaluateCandidates(single_candidate_matrix, true);

        std::ofstream test_per_instance_file((results_dir / "test_per_instance_makespans.csv").string());
        if (test_per_instance_file.is_open()) {
            test_per_instance_file << "problem_id,makespan\n";
            for (int i = 0; i < test_makespans.size(); ++i) {
                test_per_instance_file << i << "," << test_makespans[i] << "\n";
            }
            test_per_instance_file.close();
            std::cout << "[TEST] Per-instance makespans saved to test_per_instance_makespans.csv" << std::endl;
        } else {
            std::cerr << "[ERROR] Unable to open test_per_instance_makespans.csv for writing!" << std::endl;
        } */
    } catch (const std::exception& e) {
        std::cerr << "[ERROR][TEST] Exception during test evaluation: " << e.what() << std::endl;
    }

    return 0;
}