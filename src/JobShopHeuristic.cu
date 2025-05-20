#include <algorithm>
#include <cfloat>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "JobShopHeuristic.cuh"

using json = nlohmann::json;

// Constructors
JobShopHeuristic::JobShopHeuristic(const std::vector<int>& topology)
    : neuralNetwork(topology) {}

JobShopHeuristic::JobShopHeuristic(const std::string& filename)
    : neuralNetwork(InitializeNetworkFromFile(filename)) {}

JobShopHeuristic::JobShopHeuristic(NeuralNetwork&& net)
    : neuralNetwork(std::move(net)) {}

// Load neural network from file
NeuralNetwork JobShopHeuristic::InitializeNetworkFromFile(const std::string& filename) {
    std::string full_path = FileManager::GetFullPath(filename);

    if(!std::filesystem::exists(full_path)) {
        throw std::runtime_error("Network file not found: " + full_path);
	}

    std::ifstream in(full_path);
    if(!in.is_open()) {
        throw std::runtime_error("Cannot open file: " + full_path);
    }

    try {
        json j;
        in >> j;
        in.close();

        std::vector<int> loaded_topology = j["topology"];
        auto weights = j["weights"].get<std::vector<std::vector<float>>>();
        auto biases = j["biases"].get<std::vector<std::vector<float>>>();

        if(loaded_topology.empty() || weights.empty() || biases.empty()) {
            throw std::runtime_error("Invalid network data in file");
        }

        return NeuralNetwork(loaded_topology, &weights, &biases);
    } catch(const std::exception& e) {
        throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
    }
}

// Copy solution from GPU to CPU
void JobShopHeuristic::CPUSolution::FromGPU(const SolutionManager::GPUSolutions& gpuSols, int problemId) {
    int counts_offset = problemId * gpuSols.numMachines;
    int schedule_offset = problemId * gpuSols.numMachines * gpuSols.maxOps;

    std::vector<int> counts(gpuSols.numMachines);
    cudaMemcpy(counts.data(), gpuSols.allScheduleCounts + counts_offset,
               sizeof(int) * gpuSols.numMachines, cudaMemcpyDeviceToHost);

    cudaMemcpy(&makespan, gpuSols.allMakespans + problemId,
               sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<OperationSchedule> allOps(gpuSols.numMachines * gpuSols.maxOps);
    cudaMemcpy(allOps.data(), gpuSols.allSchedules + schedule_offset,
               sizeof(OperationSchedule) * allOps.size(), cudaMemcpyDeviceToHost);

    schedule.resize(gpuSols.numMachines);
    for(int m = 0; m < gpuSols.numMachines; ++m) {
        schedule[m].clear();
        for(int i = 0; i < counts[m] && i < gpuSols.maxOps; ++i) {
            int idx = m * gpuSols.maxOps + i;
            schedule[m].push_back(allOps[idx]);
        }
    }
}

// Copy solution from CPU to GPU
SolutionManager::GPUSolutions JobShopHeuristic::CPUSolution::ToGPU() const {
    SolutionManager::GPUSolutions gpuSol;
    gpuSol.numMachines = schedule.size();

    cudaMalloc(&gpuSol.allSchedules, sizeof(OperationSchedule) * schedule.size() * MAX_OPS);
    cudaMalloc(&gpuSol.allScheduleCounts, sizeof(int) * schedule.size());
    cudaMalloc(&gpuSol.allMakespans, sizeof(int));

    cudaMemcpy(gpuSol.allMakespans, &makespan, sizeof(int), cudaMemcpyHostToDevice);

    std::vector<OperationSchedule> flat_schedule;
    std::vector<int> counts;
    for(const auto& machine: schedule) {
        flat_schedule.insert(flat_schedule.end(), machine.begin(), machine.end());
        counts.push_back(machine.size());
    }

    cudaMemcpy(gpuSol.allSchedules, flat_schedule.data(),
               sizeof(OperationSchedule) * flat_schedule.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpuSol.allScheduleCounts, counts.data(),
               sizeof(int) * counts.size(),
               cudaMemcpyHostToDevice);

    return gpuSol;
}

// New solver 
void JobShopHeuristic::SolveBatchNew(
    const GPUProblem* problems,
    const NeuralNetwork::DeviceEvaluator* evaluators,
    GPUOperation* ops_working,
    float* results,
    int numProblems,
    int numWeights,
    int maxOpsPerProblem,
    cudaStream_t stream = 0
)
{
    int threads = 64;
    int blocks = numWeights;
    size_t sharedMemSize = threads * sizeof(float);

    cudaDeviceSetLimit(cudaLimitStackSize, 4096);

    int reset_value = 0;
    cudaMemcpyToSymbol(gpu_error_flag, &reset_value, sizeof(int), 0, cudaMemcpyHostToDevice);

    SolveManyWeightsKernel<<<blocks, threads, sharedMemSize, stream>>>(
        problems, evaluators, ops_working, results, numProblems, maxOpsPerProblem
    );
    cudaDeviceSynchronize();
}
// Allocate GPU memory for solutions
SolutionManager::GPUSolutions SolutionManager::CreateGPUSolutions(int numProblems, int numMachines, int maxOps) {
    GPUSolutions solutions;
    solutions.numProblems = numProblems;
    solutions.numMachines = numMachines;
    solutions.maxOps = maxOps;

    size_t schedule_size = sizeof(OperationSchedule) * numMachines * maxOps * numProblems;
    cudaMalloc(&solutions.allSchedules, schedule_size);
    cudaMemset(solutions.allSchedules, 0, schedule_size);

    size_t counts_size = numProblems * numMachines * sizeof(int);
    cudaMalloc(&solutions.allScheduleCounts, counts_size);
    cudaMemset(solutions.allScheduleCounts, 0, counts_size);

    cudaMalloc(&solutions.allMakespans, sizeof(int) * numProblems);
    cudaMemset(solutions.allMakespans, 0, numProblems * sizeof(int));

    return solutions;
}

// Free GPU memory for solutions
void SolutionManager::FreeGPUSolutions(SolutionManager::GPUSolutions& sols) {
    cudaFree(sols.allSchedules);
    cudaFree(sols.allScheduleCounts);
    cudaFree(sols.allMakespans);
    sols = GPUSolutions {};
}

// Print schedule for a solution
void JobShopHeuristic::PrintSchedule(const CPUSolution& solution, JobShopData data) {
    // Build machine->operation types map if not already available
    if(data.machineEligibleOperations.empty()) {
        data.BuildMachineEligibleOperations();
    }

    std::cout << "\n=== FINAL SCHEDULE ===" << std::endl;

    for(int machineId = 0; machineId < solution.schedule.size(); ++machineId) {
        std::cout << "M" << machineId << " (";
        bool firstOp = true;
        for(int opType: data.machineEligibleOperations[machineId]) {
            if(!firstOp) std::cout << ", ";
            std::cout << opType;
            firstOp = false;
        }
        std::cout << "): [";

        int currentTime = 0;
        bool firstElement = true;
        const auto& machineSchedule = solution.schedule[machineId];

        if(machineSchedule.empty()) {
            std::cout << "idle";
        } else {
            for(const auto& scheduledOp: machineSchedule) {
                if(scheduledOp.startTime > currentTime) {
                    if(!firstElement) std::cout << "][";
                    std::cout << "w-" << (scheduledOp.startTime - currentTime);
                    firstElement = false;
                    currentTime = scheduledOp.startTime;
                }
                if(!firstElement) std::cout << "][";
                std::cout << "t=" << scheduledOp.startTime << ",j" << scheduledOp.jobId
                          << "-o" << scheduledOp.opType
                          << "-" << (scheduledOp.endTime - scheduledOp.startTime);
                currentTime = scheduledOp.endTime;
                firstElement = false;
            }
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "Makespan: " << solution.makespan << std::endl;
}

// Update schedule after scheduling an operation (obsolete)
void JobShopHeuristic::UpdateSchedule(JobShopData& data, int jobId, int operationIdx,
                                      int machineId, CPUSolution& solution) {
    auto& job = data.jobs[jobId];
    const auto& operation = job.operations[operationIdx];

    int processingTime = data.processingTimes[operation.type][machineId];
    if(processingTime <= 0) {
        std::cerr << "Warning: Attempted to schedule zero-duration operation\n";
        return;
    }

    int machineAvailableTime = solution.schedule[machineId].empty()
                                   ? 0
                                   : solution.schedule[machineId].back().endTime;

    int startTime = std::max(machineAvailableTime, job.lastOpEndTime);
    int endTime = startTime + processingTime;

    solution.schedule[machineId].push_back({jobId, operation.type, startTime, endTime});
    job.lastOpEndTime = endTime;
    job.nextOpIndex++;
    solution.makespan = std::max(solution.makespan, endTime);
}


__global__ void SolveManyWeightsKernel(
    const GPUProblem* problems,
    const NeuralNetwork::DeviceEvaluator* evaluators,
    GPUOperation* ops_working, // Pamięć robocza, którą kernel wypełni i będzie modyfikować
    float* results,
    int numProblems,
    int maxOpsPerProblem) {

    if (gpu_error_flag) { 
        return;
    }

    extern __shared__ float shared_makespans[];

    int weightSet = blockIdx.x;
    int problemIdx = threadIdx.x; // Każdy wątek w bloku obsługuje inny problem dla danego zestawu wag

    float makespan = 0.0f; // Inicjalizacja makespan dla tego wątku/problemu

    if (problemIdx < numProblems) {
        const GPUProblem problem = problems[problemIdx]; // problem.operations wskazuje na szablon w d_ops_
                                                       // problem.totalOpsCount zawiera liczbę operacji dla tego problemu
        const NeuralNetwork::DeviceEvaluator& nn_eval = evaluators[weightSet];

        // Oblicz wskaźnik do lokalnej pamięci roboczej dla operacji tego wątku
        // To jest miejsce w ops_working, gdzie ten wątek będzie przechowywał swoją kopię operacji
        int base_working_ops_offset = (weightSet * numProblems + problemIdx) * maxOpsPerProblem;
        GPUOperation* local_ops = &ops_working[base_working_ops_offset];

        // Kopiuj operacje z szablonu (problem.operations) do lokalnej pamięci roboczej (local_ops)
        // problem.operations to wskaźnik do odpowiedniego fragmentu globalnego d_ops_ (szablonu)
        // problem.totalOpsCount to liczba operacji dla tego konkretnego problemu
        for (int i = 0; i < problem.totalOpsCount; ++i) {
            if (i < maxOpsPerProblem) { // Zabezpieczenie przed przepełnieniem bufora local_ops
                local_ops[i] = problem.operations[i]; // Kopiowanie struktury GPUOperation
            } else {
                // Opcjonalnie: obsługa błędu lub ostrzeżenie, jeśli maxOpsPerProblem jest za małe
                // Można ustawić flagę błędu lub po prostu przerwać kopiowanie
                // printf("Warning: Problem %d, WeightSet %d: totalOpsCount %d > maxOpsPerProblem %d. Truncating ops.\n", problemIdx, weightSet, problem.totalOpsCount, maxOpsPerProblem);
                break; 
            }
        }
        // Wypełnij resztę local_ops zerami lub wartościami domyślnymi, jeśli problem.totalOpsCount < maxOpsPerProblem
        // aby uniknąć używania niezainicjowanej pamięci, jeśli logika dalej zakłada pełny rozmiar maxOpsPerProblem.
        // Alternatywnie, upewnij się, że dalsza logika używa tylko problem.totalOpsCount.
        // Na razie zakładamy, że dalsza logika będzie ostrożna lub maxOpsPerProblem jest wystarczająco duże.


        const int numJobs = problem.numJobs;
        const int numMachines = problem.numMachines;

        // int base = (weightSet * numProblems + problemIdx) * maxOpsPerProblem; // Już obliczone jako base_working_ops_offset
        // GPUOperation* local_ops = &ops_working[base]; // Już zdefiniowane

        int jobScheduledOps[MAX_JOBS] = {0}; 
        int machine_times[MAX_MACHINES] = {0}; 
        
        int jobTypeCount[MAX_OP_TYPES] = {0};
        int opTypeCount[MAX_OP_TYPES] = {0};
        // int opTypePerJobCount[MAX_JOBS][MAX_OP_TYPES] = {0}; // To może być duże, rozważ optymalizację, jeśli to możliwe

        // Inicjalizacja liczników na podstawie skopiowanych local_ops
        for (int jobID = 0; jobID < numJobs; ++jobID) {
            const GPUJob& job_template = problem.jobs[jobID]; // Użyj problem.jobs do odczytu struktury joba
            jobTypeCount[job_template.type]++;
            for (int op_idx_in_job = 0; op_idx_in_job < job_template.operationCount; ++op_idx_in_job) {
                // Dostęp do operacji przez local_ops, używając offsetu z job_template
                // Pamiętaj, że job_template.operationsOffset jest teraz lokalny dla problemu (0 do N-1)
                GPUOperation& op = local_ops[job_template.operationsOffset + op_idx_in_job];
                // opTypePerJobCount[jobID][op.type]++; // Jeśli potrzebne
                opTypeCount[op.type]++;
            }
        }


        int local_makespan = 0;
        bool scheduled_any;
        do {
            scheduled_any = false;
            float bestScoreValue = -FLT_MAX;
            int bestJobID = -1, bestOpID_in_job = -1, bestMachineID = -1; // bestOpID_in_job to indeks operacji w ramach joba
            int best_local_op_idx = -1; // Globalny indeks operacji w local_ops
            int bestStartTime = 0;

            for (int jobID = 0; jobID < numJobs; ++jobID) {
                const GPUJob& current_job_template = problem.jobs[jobID]; // Odczyt z szablonu problemu
                if (jobScheduledOps[jobID] == current_job_template.operationCount)
                    continue;

                for (int op_idx_in_job = 0; op_idx_in_job < current_job_template.operationCount; ++op_idx_in_job) {
                    // Dostęp do operacji przez local_ops
                    int current_local_op_idx = current_job_template.operationsOffset + op_idx_in_job;
                    GPUOperation& operation = local_ops[current_local_op_idx]; 
                    
                    if (operation.predecessorCount != 0) continue; // Jeśli ma niespełnione zależności

                    for (int m_idx = 0; m_idx < operation.eligibleCount; m_idx++) {
                        int machineID = problem.eligibleMachines[operation.eligibleMachinesOffset + m_idx];
                        int start_time = max(machine_times[machineID], operation.lastPredecessorEndTime);
                        int opMach_idx = operation.type * numMachines + machineID; // Indeks w spłaszczonej tablicy czasów przetwarzania
                        int pTime = problem.processingTimes[opMach_idx];

                        if (pTime <= 0) continue; // Pomiń, jeśli czas przetwarzania jest nieprawidłowy

                        float features[1 + 2 * MAX_MACHINES + 3 * MAX_OP_TYPES + 2 * MAX_JOB_TYPES] = {0.0f};
                        // ... (wypełnianie features - bez zmian) ...
                        features[0] = static_cast<float>(start_time) - machine_times[machineID];
                        for (int i = 1; i < MAX_MACHINES + 1; ++i) {
                            features[i] = static_cast<float>(local_makespan - machine_times[i - 1]);
                        }
                        features[1 + machineID] = static_cast<float>(local_makespan - (start_time + pTime));
                        features[1 + MAX_MACHINES + machineID] = 1.0f;
                        features[1 + 2 * MAX_MACHINES + operation.type] = 1.0f;
                        
                        const float SCALE_FACTOR = 100.0f;
                        features[0] /= SCALE_FACTOR;
                        for (int i = 1; i < MAX_MACHINES + 1; ++i) {
                            features[i] /= SCALE_FACTOR;
                        }
                        features[1 + machineID] /= SCALE_FACTOR;

                        float score = nn_eval.Evaluate(features);

                        if (score > bestScoreValue) {
                            bestScoreValue = score;
                            bestJobID = jobID;
                            bestOpID_in_job = op_idx_in_job; // Zapisz indeks operacji w ramach joba
                            best_local_op_idx = current_local_op_idx; // Zapisz globalny indeks w local_ops
                            bestMachineID = machineID;
                            bestStartTime = start_time;
                        }
                    }
                }
            }

            if (bestJobID == -1) break; // Nie znaleziono żadnej operacji do uszeregowania

            // const GPUJob& bestJob_template = problem.jobs[bestJobID]; // Odczyt z szablonu
            // GPUOperation& bestOperation = local_ops[bestJob_template.operationsOffset + bestOpID_in_job]; // Dostęp przez local_ops
            GPUOperation& bestOperation = local_ops[best_local_op_idx]; // Użyj zapisanego globalnego indeksu

            int opMach_idx = bestOperation.type * numMachines + bestMachineID;
            int pTime = problem.processingTimes[opMach_idx];
            int endTime = bestStartTime + pTime;

            jobScheduledOps[bestJobID]++;
            opTypeCount[bestOperation.type]--; // Zaktualizuj globalny licznik typów operacji

            const GPUJob& scheduled_job_template = problem.jobs[bestJobID]; // Potrzebne do odczytu typu joba
            if (jobScheduledOps[bestJobID] == scheduled_job_template.operationCount) {
                jobTypeCount[scheduled_job_template.type]--;
            }

            bestOperation.predecessorCount = -1; // Oznacz jako uszeregowaną (lub użyj innej flagi)
                                                 // Uważaj, jeśli -1 ma inne znaczenie. Lepsza byłaby dedykowana flaga bool.

            // Aktualizuj zależności dla następników
            for (int s_idx = 0; s_idx < bestOperation.successorCount; ++s_idx) {
                // successorID_in_job to indeks następnika W RAMACH TEGO SAMEGO JOBA
                int successorID_in_job = problem.successorsIDs[bestOperation.successorsOffset + s_idx];
                // Oblicz globalny indeks następnika w local_ops
                int successor_local_op_idx = scheduled_job_template.operationsOffset + successorID_in_job;
                GPUOperation& successorOperation = local_ops[successor_local_op_idx];
                
                successorOperation.predecessorCount -= 1;
                successorOperation.lastPredecessorEndTime = max(successorOperation.lastPredecessorEndTime, endTime);
            }

            machine_times[bestMachineID] = endTime;
            if (endTime > local_makespan) local_makespan = endTime;

            scheduled_any = true;
        } while (scheduled_any);

        makespan = static_cast<float>(local_makespan);
        shared_makespans[problemIdx] = makespan;

    } else { // problemIdx >= numProblems
        shared_makespans[problemIdx] = 0.0f; // Wątki poza zakresem problemów
    }
    __syncthreads(); // Synchronizuj wszystkie wątki w bloku przed obliczeniem średniej

    // Oblicz średni makespan dla zestawu wag (tylko wątek 0 w bloku)
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        int valid_problems_count = 0;
        for (int i = 0; i < numProblems; ++i) { // Sumuj tylko dla aktywnych problemów
            if (shared_makespans[i] >= 0) { // Lub inna walidacja, jeśli 0 jest poprawnym makespanem
                 sum += shared_makespans[i];
                 valid_problems_count++;
            }
        }
        if (valid_problems_count > 0) {
            results[weightSet] = sum / valid_problems_count;
        } else {
            results[weightSet] = FLT_MAX; // Lub inna wartość błędu, jeśli żaden problem nie został rozwiązany
        }
    }
}
// Print problem details from device (for debugging)
__device__ void PrintProblemDetails(const GPUProblem& problem) {
    printf("\n=== Problem %d Details ===\n", blockIdx.x * blockDim.x + threadIdx.x);
    printf("Machines: %d, Jobs: %d, Operation Types: %d\n",
           problem.numMachines, problem.numJobs, problem.numOpTypes);

    printf("\nJobs:\n");
    for(int j = 0; j < problem.numJobs; j++) {
        GPUJob job = problem.jobs[j];
        printf("Job %d, of type %d (%d ops):\n", job.id, job.type, job.operationCount);

        for(int o = 0; o < job.operationCount; o++) {
            GPUOperation op = problem.operations[job.operationsOffset + o];
            printf("  Op type %d on machines: ", op.type);
        
            for(int m = 0; m < op.eligibleCount; m++) {
                printf("%d ", problem.eligibleMachines[op.eligibleMachinesOffset + m]);
            }
            printf("\n");
        }
    }
    printf("========================\n\n");
}