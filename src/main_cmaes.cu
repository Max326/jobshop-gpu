#include "JobShopGPUEvaluator.cuh"
#include "NeuralNetwork.cuh"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <filesystem> // Required for std::filesystem
#include <nlohmann/json.hpp> // For JSON operations
#include <random>
#include <algorithm>

// Assuming JobShopGPUEvaluator and NeuralNetwork classes are defined elsewhere
// and g_gpu_evaluator is correctly handled.
JobShopGPUEvaluator* g_gpu_evaluator = nullptr;

// Function to generate test weights (as provided in your example)
Eigen::MatrixXd GenerateTestWeights(int params, int candidates, float min_val = -5.0f, float max_val = 5.0f) {
    Eigen::MatrixXd weights(params, candidates);
    // Using a fixed seed for reproducibility, change if varied randomness is needed each run
    std::default_random_engine generator(std::random_device{}()); 
    std::normal_distribution<float> dist(0.0, 0.5); 
    
    for(int c = 0; c < candidates; ++c) {
        for(int p = 0; p < params; ++p) {
            float val = dist(generator);
            // Clamp values to prevent numerical instability
            weights(p, c) = std::max(min_val, std::min(max_val, val));
        }
    }
    return weights;
}

// Function to save Eigen::MatrixXd to a JSON file
bool SaveMatrixToJson(const Eigen::MatrixXd& matrix, const std::string& filename) {
    // Construct path to project_root/data/filename
    std::filesystem::path data_dir_path = std::filesystem::path(".."); 
    data_dir_path /= "data"; // Now data_dir_path is ../data

    try {
        // Ensure the directory exists; creates it if it doesn't.
        if (!std::filesystem::exists(data_dir_path)) {
            std::filesystem::create_directories(data_dir_path);
            std::cout << "Created directory: " << std::filesystem::absolute(data_dir_path) << std::endl;
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error when creating directory: " << e.what() << std::endl;
        return false;
    }

    std::string full_path = (data_dir_path / filename).string();
    
    nlohmann::json j_matrix = nlohmann::json::array();
    
    for (int col_idx = 0; col_idx < matrix.cols(); ++col_idx) {
        nlohmann::json j_col = nlohmann::json::array();
        for (int row_idx = 0; row_idx < matrix.rows(); ++row_idx) {
            j_col.push_back(matrix(row_idx, col_idx));
        }
        j_matrix.push_back(j_col);
    }
    
    std::ofstream out_stream(full_path);
    if (!out_stream) {
        std::cerr << "Failed to open file for writing: " << full_path << std::endl;
        return false;
    }
    
    out_stream << j_matrix.dump(4); // Use dump(4) for pretty printing
    std::cout << "Matrix saved to: " << std::filesystem::absolute(full_path) << std::endl;
    return true;
}

// Function to load Eigen::MatrixXd from a JSON file
bool LoadMatrixFromJson(Eigen::MatrixXd& matrix, const std::string& filename) {
    // Construct path to project_root/data/filename
    std::filesystem::path data_dir_path = std::filesystem::path("..");
    data_dir_path /= "data"; // Now data_dir_path is ../data
    std::string full_path = (data_dir_path / filename).string();
    
    if (!std::filesystem::exists(full_path)) {
        std::cerr << "File not found: " << full_path << std::endl;
        return false;
    }
    
    std::ifstream in_stream(full_path);
    if (!in_stream) {
        std::cerr << "Failed to open file for reading: " << full_path << std::endl;
        return false;
    }
    
    nlohmann::json j_matrix;
    try {
        in_stream >> j_matrix;
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "Failed to parse JSON from file " << full_path << ": " << e.what() << std::endl;
        return false;
    }
    
    if (!j_matrix.is_array() || j_matrix.empty() || !j_matrix[0].is_array()) {
        std::cerr << "Invalid JSON format for matrix in file " << full_path << std::endl;
        return false;
    }
    
    int num_cols = j_matrix.size();
    int num_rows = j_matrix[0].size();
    
    matrix.resize(num_rows, num_cols); // Resize the matrix to fit the loaded data
    
    for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
        if (!j_matrix[col_idx].is_array() || j_matrix[col_idx].size() != num_rows) {
            std::cerr << "Inconsistent column size in JSON data in file " << full_path << std::endl;
            return false;
        }
        for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
            matrix(row_idx, col_idx) = j_matrix[col_idx][row_idx].get<double>(); // Or float, depending on Eigen::MatrixXd type
        }
    }
    
    std::cout << "Loaded matrix from: " << std::filesystem::absolute(full_path) << std::endl;
    return true;
}

int main(int argc, char *argv[]) {
    const std::vector<int> topology = {86, 32, 16, 1};
    const int batch_size = 50;
    const std::string problem_file = "test_10k.json"; // This seems to be an input file, ensure its path is correct
    const int population_size = 192;

    const std::string weights_filename = "nn_weights.json"; // Filename for weights
    bool load_weights_from_file = true; 

    // Example: Simple command-line argument parsing to set the flag
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--load-weights") {
            load_weights_from_file = true;
            break;
        }
    }

    try {
        JobShopGPUEvaluator gpu_evaluator(problem_file, topology, population_size); // Assuming problem_file path is handled by JobShopGPUEvaluator
        g_gpu_evaluator = &gpu_evaluator;
        
        // 2. Generate or load test weights
        const int nn_weights_count = NeuralNetwork::CalculateTotalParameters(topology);
        Eigen::MatrixXd candidates(nn_weights_count, population_size);
        
        std::filesystem::path target_weights_file_path = std::filesystem::path("..") / "data" / weights_filename;

        if (load_weights_from_file && std::filesystem::exists(target_weights_file_path)) {
            std::cout << "[INFO] Attempting to load weights from " << std::filesystem::absolute(target_weights_file_path) << std::endl;
            if (LoadMatrixFromJson(candidates, weights_filename)) {
                if (candidates.rows() != nn_weights_count || candidates.cols() != population_size) {
                    std::cout << "[WARNING] Loaded matrix dimensions (" << candidates.rows() << "x" 
                              << candidates.cols() << ") do not match expected dimensions ("
                              << nn_weights_count << "x" << population_size 
                              << "). Generating new weights instead." << std::endl;
                    candidates = GenerateTestWeights(nn_weights_count, population_size);
                    SaveMatrixToJson(candidates, weights_filename); // Save the newly generated weights
                } else {
                    std::cout << "[INFO] Successfully loaded weights from file." << std::endl;
                }
            } else {
                std::cout << "[WARNING] Failed to load weights from file. Generating new weights." << std::endl;
                candidates = GenerateTestWeights(nn_weights_count, population_size);
                SaveMatrixToJson(candidates, weights_filename); // Save the newly generated weights
            }
        } else {
            if (load_weights_from_file) {
                 std::cout << "[INFO] Weights file not found at " << std::filesystem::absolute(target_weights_file_path) << ". Generating new weights." << std::endl;
            } else {
                std::cout << "[INFO] Generating new weights (load_weights_from_file is false)." << std::endl;
            }
            candidates = GenerateTestWeights(nn_weights_count, population_size);
            SaveMatrixToJson(candidates, weights_filename);
        }      
        
        // Validate generated/loaded weights for NaN values (example check on first candidate)
        for (int p = 0; p < nn_weights_count; ++p) {
            if (candidates.cols() > 0 && std::isnan(candidates(p, 0))) {
                std::cerr << "[ERROR] NaN detected in weights at param " << p 
                          << " for the first candidate." << std::endl;
                return 1; // Exit if NaN is found
            }
        }        
        
        std::cout << "[DIAG] Using " << candidates.cols() << " weight sets with "
                  << candidates.rows() << " parameters each.\n";

        // 3. Evaluate single batch (example)
        // for (int i = 0; i < 3; ++i){
        //     std::cout << "[INFO] Evaluating batch " << i << "...\n";
        //     if (gpu_evaluator.SetCurrentBatch(i * batch_size, batch_size)) {
        //         Eigen::VectorXd results = gpu_evaluator.EvaluateCandidates(candidates);
        //     } else {
        //         std::cerr << "[ERROR] Failed to set evaluation batch\n";
        //         return 1;
        //     }
        // }

        if (gpu_evaluator.SetCurrentBatch(0, batch_size)) {
            Eigen::VectorXd results = gpu_evaluator.EvaluateCandidates(candidates);
        } else {
            std::cerr << "Failed to set evaluation batch\n";
            return 1;
        }
    
    } catch(const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

