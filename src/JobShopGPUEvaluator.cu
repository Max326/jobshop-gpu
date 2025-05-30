#include "JobShopGPUEvaluator.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <chrono>

JobShopGPUEvaluator::JobShopGPUEvaluator(const std::string& problem_file, const std::vector<int>& nn_topology, const int &population_size)
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

    // Initialize DeviceEvaluator pool:
    nn_candidate_count_ = population_size; // number of candidates, use the value you are using in the CMAES
    neural_networks_.resize(nn_candidate_count_);
    host_evaluators_.resize(nn_candidate_count_);

    nn_total_weights_per_network_ = 0;
    nn_total_biases_per_network_ = 0;
    for (size_t i = 1; i < nn_topology_.size(); ++i) {
        nn_total_weights_per_network_ += nn_topology_[i - 1] * nn_topology_[i];
        nn_total_biases_per_network_ += nn_topology_[i];
    }

    total_weights_size_ = (size_t)nn_candidate_count_ * nn_total_weights_per_network_ * sizeof(float);
    total_biases_size_ = (size_t)nn_candidate_count_ * nn_total_biases_per_network_ * sizeof(float);

    // Allocate pinned host memory
    CUDA_CHECK(cudaHostAlloc(&h_pinned_all_weights_, total_weights_size_, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_pinned_all_biases_, total_biases_size_, cudaHostAllocDefault));

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_all_candidate_weights_, total_weights_size_));
    CUDA_CHECK(cudaMalloc(&d_all_candidate_biases_, total_biases_size_));

    for (int r = 0; r < nn_candidate_count_; ++r) {
        neural_networks_[r] = NeuralNetwork(nn_topology_); // Tworzy NN. Domyślnie manage_gpu_buffers = true,
                                                           // więc NN alokuje swoje d_weights/d_biases.
        
        // Ponieważ będziemy używać współdzielonych pul, musimy zwolnić te właśnie zaalokowane przez NN.
        if (neural_networks_[r].cudaData) { // Sprawdź czy cudaData istnieje
            if (neural_networks_[r].cudaData->d_weights) {
                CUDA_CHECK(cudaFree(neural_networks_[r].cudaData->d_weights));
                neural_networks_[r].cudaData->d_weights = nullptr; // Ustaw na null po zwolnieniu
            }
            if (neural_networks_[r].cudaData->d_biases) {
                CUDA_CHECK(cudaFree(neural_networks_[r].cudaData->d_biases));
                neural_networks_[r].cudaData->d_biases = nullptr; // Ustaw na null po zwolnieniu
            }

            // Ustaw wskaźniki na współdzielone, prealokowane pule
            neural_networks_[r].cudaData->d_weights = d_all_candidate_weights_ + r * nn_total_weights_per_network_;
            neural_networks_[r].cudaData->d_biases = d_all_candidate_biases_ + r * nn_total_biases_per_network_;
            
            // Poinformuj instancję NN, że nie powinna zwalniać tych współdzielonych wskaźników
            neural_networks_[r].cudaData->manage_gpu_buffers = false; 
        } else {
            // To nie powinno się zdarzyć, jeśli konstruktor NN działa poprawnie
            throw std::runtime_error("JobShopGPUEvaluator: neural_networks_[" + std::to_string(r) + "].cudaData is null after construction.");
        }
        
        host_evaluators_[r] = neural_networks_[r].GetDeviceEvaluator(); 
    }    

    // Allocate and copy DeviceEvaluators to GPU
    cudaMalloc(&d_evaluators_, sizeof(NeuralNetwork::DeviceEvaluator) * nn_candidate_count_);
    cudaMemcpy(d_evaluators_, host_evaluators_.data(), sizeof(NeuralNetwork::DeviceEvaluator) * nn_candidate_count_, cudaMemcpyHostToDevice);
    d_ops_working_pool_ = nullptr; 
    d_template_ops_offsets_ = nullptr;
}

JobShopGPUEvaluator::~JobShopGPUEvaluator() {
    FreeProblemDataGPU();
    cudaFree(d_evaluators_);
    cudaFree(d_all_candidate_weights_);
    cudaFree(d_all_candidate_biases_);
    cudaFreeHost(h_pinned_all_weights_);  // Use cudaFreeHost for pinned memory
    cudaFreeHost(h_pinned_all_biases_);  // Use cudaFreeHost for pinned memory

    if (d_ops_working_pool_) {
        cudaFree(d_ops_working_pool_);
        d_ops_working_pool_ = nullptr;
    }
}

void JobShopGPUEvaluator::FreeProblemDataGPU() {
    JobShopDataGPU::FreeBatchGPUData(d_problems_, d_jobs_, d_ops_, d_eligible_, d_succ_, d_procTimes_);
    d_problems_ = nullptr;
    d_jobs_ = nullptr;
    d_ops_ = nullptr;
    d_eligible_ = nullptr;
    d_succ_ = nullptr;
    d_procTimes_ = nullptr;

    if (d_template_ops_offsets_) {
        cudaFree(d_template_ops_offsets_);
        d_template_ops_offsets_ = nullptr;
    }
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

    // NOWE: Alokacja i kopiowanie d_template_ops_offsets_
    if (num_problems_to_evaluate_ > 0) {
        // cpu_batch_data_.operationsOffsets ma rozmiar num_problems_to_evaluate_ + 1
        CUDA_CHECK(cudaMalloc(&d_template_ops_offsets_, (num_problems_to_evaluate_ + 1) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_template_ops_offsets_, cpu_batch_data_.operationsOffsets.data(), (num_problems_to_evaluate_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
    }

    // NOWE: Alokacja d_ops_working_pool_ (lub realokacja jeśli rozmiar się zmienił)
    if (d_ops_working_pool_) { // Zwolnij stary, jeśli istnieje
        cudaFree(d_ops_working_pool_);
        d_ops_working_pool_ = nullptr;
    }
    if (nn_candidate_count_ > 0 && num_problems_to_evaluate_ > 0 && max_ops_per_problem_ > 0) {
        size_t pool_size = (size_t)nn_candidate_count_ * num_problems_to_evaluate_ * max_ops_per_problem_ * sizeof(GPUOperation);
        CUDA_CHECK(cudaMalloc(&d_ops_working_pool_, pool_size));
    }
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
    auto t_total_start = std::chrono::high_resolution_clock::now();

    int current_nn_candidate_count = candidates.cols();
    if (candidates.rows() != nn_total_params_) {
        throw std::runtime_error("EvaluateCandidates: Mismatch in number of weights per NN candidate.");
    }
    if (current_nn_candidate_count != this->nn_candidate_count_) {
        // Rozważ, czy to jest błąd, czy dynamiczna zmiana rozmiaru populacji jest dozwolona.
        // Jeśli dozwolona, bufory zależne od nn_candidate_count_ (np. d_ops_working_pool_)
        // mogą wymagać realokacji lub obsługi tego przypadku.
        // Na razie zakładamy, że nn_candidate_count_ jest stałe po inicjalizacji.
        // Jeśli current_nn_candidate_count może być mniejsze, to OK, ale jeśli większe, to problem.
        std::cout << "[WARNING] EvaluateCandidates: current_nn_candidate_count (" << current_nn_candidate_count 
                  << ") differs from member nn_candidate_count_ (" << this->nn_candidate_count_ << ")." << std::endl;
        // Można rzucić błąd lub dostosować this->nn_candidate_count_ i realokować bufory,
        // ale to wykracza poza zakres tej funkcji.
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // --- Sekcja aktualizacji wag i DeviceEvaluators ---
    auto t_weight_update_start = std::chrono::high_resolution_clock::now();

    // 1. Wypełnij przypiętą pamięć hosta
    for (int r = 0; r < current_nn_candidate_count; ++r) {
        int paramIdx = 0;
        // Używaj this->nn_total_weights_per_network_ i this->nn_total_biases_per_network_
        // które są obliczane na podstawie this->nn_topology_
        size_t weight_buffer_offset = (size_t)r * this->nn_total_weights_per_network_;
        size_t bias_buffer_offset = (size_t)r * this->nn_total_biases_per_network_;

        for (size_t i = 1; i < this->nn_topology_.size(); ++i) {
            int prevLayerSize = this->nn_topology_[i - 1];
            int currLayerSize = this->nn_topology_[i];

            for (int w_idx = 0; w_idx < prevLayerSize * currLayerSize; ++w_idx) {
                if (weight_buffer_offset + w_idx < this->total_weights_size_ / sizeof(float)) { // Sprawdzenie granic
                    h_pinned_all_weights_[weight_buffer_offset + w_idx] = static_cast<float>(candidates(paramIdx++, r));
                } else { /* Obsługa błędu przekrocenia bufora */ }
            }
            weight_buffer_offset += prevLayerSize * currLayerSize; // Przesuń główny offset dla następnej warstwy w następnej iteracji 'r'

            for (int b_idx = 0; b_idx < currLayerSize; ++b_idx) {
                 if (bias_buffer_offset + b_idx < this->total_biases_size_ / sizeof(float)) { // Sprawdzenie granic
                    h_pinned_all_biases_[bias_buffer_offset + b_idx] = static_cast<float>(candidates(paramIdx++, r));
                 } else { /* Obsługa błędu przekrocenia bufora */ }
            }
            bias_buffer_offset += currLayerSize; // Przesuń główny offset dla następnej warstwy w następnej iteracji 'r'
        }
    }
    
    auto t_pinned_mem_populated = std::chrono::high_resolution_clock::now();

    // 2. Asynchroniczny transfer wag i biasów
    CUDA_CHECK(cudaMemcpyAsync(d_all_candidate_weights_, h_pinned_all_weights_, total_weights_size_, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_all_candidate_biases_, h_pinned_all_biases_, total_biases_size_, cudaMemcpyHostToDevice, stream));
    // Nie ma potrzeby synchronizacji tutaj, jeśli następne operacje na tych danych są również w tym strumieniu

    // 3. Aktualizacja host_evaluators_ i transfer do d_evaluators_
    // Zakładając, że poprawka zarządzania pamięcią NeuralNetwork jest wdrożona:
    for (int r = 0; r < current_nn_candidate_count; ++r) {
        // Wskaźniki d_weights/d_biases w neural_networks_[r].cudaData są już ustawione w konstruktorze
        // i wskazują na odpowiednie fragmenty d_all_candidate_weights_/biases.
        // Flaga manage_gpu_buffers powinna być false dla tych instancji.
        host_evaluators_[r] = neural_networks_[r].GetDeviceEvaluator();
    }
    CUDA_CHECK(cudaMemcpyAsync(d_evaluators_, host_evaluators_.data(), sizeof(NeuralNetwork::DeviceEvaluator) * current_nn_candidate_count, cudaMemcpyHostToDevice, stream));
    
    auto t_evaluators_updated = std::chrono::high_resolution_clock::now();

    // --- Sekcja przygotowania d_ops_working_pool_ na GPU ---
    if (d_ops_working_pool_ && d_ops_ && d_template_ops_offsets_ && current_nn_candidate_count > 0 && num_problems_to_evaluate_ > 0) {
        int threadsPerBlockInit = 256;
        int blocksPerGridInit = (current_nn_candidate_count + threadsPerBlockInit - 1) / threadsPerBlockInit;
        
        InitializeWorkingOpsKernel<<<blocksPerGridInit, threadsPerBlockInit, 0, stream>>>(
            d_ops_working_pool_,
            d_ops_,
            d_template_ops_offsets_,
            current_nn_candidate_count,
            num_problems_to_evaluate_,
            max_ops_per_problem_
        );
        cudaError_t initKernelErr = cudaGetLastError(); // Sprawdź błąd po kernelu
        if (initKernelErr != cudaSuccess) {
            std::cerr << "EvaluateCandidates: InitializeWorkingOpsKernel error: " << cudaGetErrorString(initKernelErr) << std::endl;
            CUDA_CHECK(cudaStreamDestroy(stream));
            return Eigen::VectorXd::Constant(current_nn_candidate_count, 1e9 + 1); // Inna wartość błędu
        }
    } else if (current_nn_candidate_count > 0 && num_problems_to_evaluate_ > 0) {
        std::cerr << "[WARNING] EvaluateCandidates: Skipping InitializeWorkingOpsKernel due to null pointers or zero sizes for essential data." << std::endl;
        // To może być krytyczny błąd konfiguracji, jeśli oczekiwano wykonania kernela.
    }
    
    auto t_ops_working_prepared = std::chrono::high_resolution_clock::now();

    // --- Sekcja uruchomienia głównego kernela i pobrania wyników ---
    float* d_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(float) * current_nn_candidate_count));

    auto t_kernel_launch_prep = std::chrono::high_resolution_clock::now();

    JobShopHeuristic::SolveBatchNew(
        d_problems_, 
        d_evaluators_, 
        d_ops_working_pool_, // Użyj prealokowanego bufora
        d_results,
        num_problems_to_evaluate_,
        current_nn_candidate_count,
        max_ops_per_problem_,
        stream,
        nn_total_params_ // Upewnij się, że to jest nn_total_params_ *dla jednej sieci*, a nie suma dla wszystkich
    );
    
    cudaError_t mainKernelErr = cudaGetLastError(); // Sprawdź błąd po kernelu, przed synchronizacją
    if (mainKernelErr != cudaSuccess) {
        std::cerr << "EvaluateCandidates: SolveBatchNew kernel launch error: " << cudaGetErrorString(mainKernelErr) << std::endl;
        CUDA_CHECK(cudaFree(d_results));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return Eigen::VectorXd::Constant(current_nn_candidate_count, 1e9 + 2); // Inna wartość błędu
    }

    CUDA_CHECK(cudaStreamSynchronize(stream)); // Synchronizuj po wszystkich operacjach w strumieniu
    
    auto t_kernel_finished = std::chrono::high_resolution_clock::now();

    std::vector<float> host_results(current_nn_candidate_count);
    CUDA_CHECK(cudaMemcpy(host_results.data(), d_results, sizeof(float) * current_nn_candidate_count, cudaMemcpyDeviceToHost)); // Kopiowanie synchroniczne jest OK po synchronizacji strumienia

    auto t_results_copied_d2h = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd fvalues(current_nn_candidate_count);
    for (int r = 0; r < current_nn_candidate_count; ++r) {
        fvalues[r] = static_cast<double>(host_results[r]);
    }
    
    auto t_fvalues_filled = std::chrono::high_resolution_clock::now();

    double min_makespan = (fvalues.size() > 0) ? fvalues.minCoeff() : 0.0;
    if (fvalues.size() > 0) { // Drukuj tylko jeśli są wyniki
        std::cout << "[INFO] Best average makespan in batch: " << min_makespan << std::endl;
    }

    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "[TIMER][EvaluateCandidates] "
              << "WeightUpdateToPinned: " << std::chrono::duration_cast<std::chrono::microseconds>(t_pinned_mem_populated - t_weight_update_start).count() << " us, "
              << "HostToDeviceAsync (Weights, Biases, Evaluators): " << std::chrono::duration_cast<std::chrono::microseconds>(t_evaluators_updated - t_pinned_mem_populated).count() << " us, "
              << "InitWorkingOpsKernel: " << std::chrono::duration_cast<std::chrono::microseconds>(t_ops_working_prepared - t_evaluators_updated).count() << " us, "
              << "SolveBatchNewKernel (incl. sync): " << std::chrono::duration_cast<std::chrono::microseconds>(t_kernel_finished - t_kernel_launch_prep).count() << " us, "
              // t_kernel_launch_prep jest po alokacji d_results, więc t_ops_working_prepared do t_kernel_launch_prep to czas alokacji d_results
              << "ResultsD2H: " << std::chrono::duration_cast<std::chrono::microseconds>(t_results_copied_d2h - t_kernel_finished).count() << " us, "
              << "FValuesFill: " << std::chrono::duration_cast<std::chrono::microseconds>(t_fvalues_filled - t_results_copied_d2h).count() << " us, "
              << "TOTAL: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_fvalues_filled - t_total_start).count() << " ms"
              << std::endl;

    return fvalues;
}


