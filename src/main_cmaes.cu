#include "customCMAES.hpp"
#include "JobShopGPUEvaluator.cuh"
#include <fstream> // Dodaj na górze pliku

/*
:0 will be used to mark OG Stanislaus's code regarding cmaes, 
so that i (or Copilot) dont fuck it up 
*/
using namespace libcmaes;

JobShopGPUEvaluator* g_gpu_train_evaluator = nullptr;
JobShopGPUEvaluator* g_gpu_validate_evaluator = nullptr;

float best_val_makespan = std::numeric_limits<float>::max();
Eigen::VectorXd best_weights; // Will hold the best weights found so far

int main(int argc, char *argv[])
{
    const std::vector<int> topology = {86, 32, 16, 1};
    const int batch_size = 50;
    const int train_problem_count = 130000; //130k
    const int validation_problem_count = 10000; // TODO change to 10k!!! when file is ready

    // --- Nowe pliki ---
    const std::string train_problem_file = "TRAIN/rnd_JT(5)_J(15)_M(5)_JO(5-10)_O(20)_OM(1-3)_total.json"; // used to be test_problem_file
    const std::string validate_problem_file = "VALID/rnd_JT(5)_J(15)_M(5)_JO(5-10)_O(20)_OM(1-3)_validation.json";
    const std::string test_problem_file = "TEST/rnd_JT(5)_J(15)_M(5)_JO(5-10)_O(20)_OM(1-3)_test.json"; 

    int population_size = 192;//:0

    int nn_weights_count = NeuralNetwork::CalculateTotalParameters(topology);//:0
    
    std::cout<< nn_weights_count <<std::endl;
    
    JobShopGPUEvaluator gpu_evaluator(train_problem_file, topology, population_size, train_problem_count);
    g_gpu_train_evaluator = &gpu_evaluator;

    g_gpu_validate_evaluator = new JobShopGPUEvaluator(validate_problem_file, topology, population_size, validation_problem_count);

    // int total_problems = gpu_evaluator.GetTotalProblems();
    std::vector<double> x0(nn_weights_count, 0.0);
    /*     for(int i = 0; i < nn_weights_count; i++) {
        x0[i] = (double)rand() / RAND_MAX * 0.01 - 0.005;
        
    } */
    
    double sigma = 0.1;//:0
    CMAParameters<> cmaparams(x0, sigma, population_size);//:0  
    cmaparams.set_sep();
    cmaparams.set_algo(sepaCMAES);
    
    FitFunc eval = [](const double *x, const int N) -> double { return 0.0; }; //:0
    ESOptimizer<customCMAStrategy,CMAParameters<>> optim(eval, cmaparams);//:0

    // --- Dodaj pliki do zapisu najlepszych makespanów ---
    std::ofstream train_makespan_file("best_train_makespans.csv");
    std::ofstream val_makespan_file("best_val_makespans.csv");
    train_makespan_file << "iteration,best_train_makespan\n";
    val_makespan_file << "iteration,best_val_makespan\n";
    // ----------------------------------------------------

    int batch_start = 0;
    int global_iter=0;

    while(!optim.stop() && gpu_evaluator.SetCurrentBatch(batch_start, batch_size)) {
        dMat candidates = optim.ask();//:0
        optim.eval(candidates);//:0
        optim.tell();//:0
        optim.inc_iter();//:0

        // best_val_makespan = min(best_val_makespan, optim.get_best_fvalue()); //?
        std::cout << "Best makespan: " << optim.get_best_fvalue() << std::endl;

        // --- Zapisuj najlepszy makespan z treningu po każdej iteracji ---
        train_makespan_file << global_iter << "," << optim.get_best_fvalue() << "\n";
        train_makespan_file.flush();
        // ---------------------------------------------------------------

        batch_start += batch_size;
        global_iter++;
        
        std::cout << "Global iterations: " << global_iter << std::endl;

        if (global_iter % 10 == 0) {
            // 1. Get the best weights from the current training population
            const auto& cma_weights = optim.get_solutions().best_candidate().get_x();
            Eigen::VectorXd current_best_weights = Eigen::Map<const Eigen::VectorXd>(cma_weights.data(), cma_weights.size());

            // 2. Evaluate this candidate on 10,000 problems to get the lowest makespan
            std::cout << "[VALIDATION] Iter " << global_iter 
                    << ": Running validation on " << validation_problem_count << " problems..." << std::endl;

            // const int validation_batch_size = validation_problem_count / population_size;
                    
            float lowest_makespan = g_gpu_validate_evaluator->EvaluateForMinMakespan(current_best_weights, validation_problem_count);

            std::cout << "[VALIDATION] Lowest makespan found = " << lowest_makespan
                    << " (best so far: " << best_val_makespan << ")" << std::endl;

            // --- Zapisuj najlepszy makespan z walidacji po każdej iteracji walidującej ---
            val_makespan_file << global_iter << "," << lowest_makespan << "\n";
            val_makespan_file.flush();
            // ------------------------------------------------------------------------------

            // 3. If it's a new global best, save the weights
            if (lowest_makespan < best_val_makespan) {
                std::cout << "[VALIDATION] New best network found!" << std::endl;
                best_val_makespan = lowest_makespan;
                best_weights = current_best_weights;

                // --- Your existing file-saving code ---
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
                // --- End of file-saving code ---
            }
        }
        
    }

    // --- Zamknij pliki na końcu programu ---
    train_makespan_file.close();
    val_makespan_file.close();
    // ----------------------------------------

    // --- TEST NA ZBIORZE TESTOWYM ---
    try {
        const int test_problem_count = 100; // lub inna liczba, jeśli masz inny rozmiar testu

        JobShopGPUEvaluator test_evaluator(test_problem_file, topology, population_size, test_problem_count);

        std::cout << "[TEST] Evaluating best weights on test set (" << test_problem_count << " problems)..." << std::endl;
        float test_makespan = test_evaluator.EvaluateForMinMakespan(best_weights, test_problem_count);

        // Zapisz wynik do pliku
        std::ofstream test_result_file("best_test_result.csv");
        if (test_result_file.is_open()) {
            test_result_file << "makespan," << test_makespan << "\n";
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
    } catch (const std::exception& e) {
        std::cerr << "[ERROR][TEST] Exception during test evaluation: " << e.what() << std::endl;
    }
    // --- KONIEC TESTU ---

    return 0;
}