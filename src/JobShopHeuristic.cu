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
    int maxOpsPerProblem 
)
{
    int threads = 64;
    int blocks = numWeights;
    size_t sharedMemSize = threads * sizeof(float);

    cudaDeviceSetLimit(cudaLimitStackSize, 4096);

    SolveManyWeightsKernel<<<blocks, threads, sharedMemSize>>>(
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
    GPUOperation* ops_working,
    float* results,
    int numProblems,
    int maxOpsPerProblem) {
    extern __shared__ float shared_makespans[];

    int weightSet = blockIdx.x;
    int problemIdx = threadIdx.x;

    // Debug: Start kernela
    if (weightSet == 0 && problemIdx == 0) {
        printf("[KERNEL] SolveManyWeightsKernel start: numProblems=%d, maxOpsPerProblem=%d\n", numProblems, maxOpsPerProblem);
    }

    float makespan = 0.0f;

    if (problemIdx < numProblems) {
        const GPUProblem problem = problems[problemIdx];
        const NeuralNetwork::DeviceEvaluator& nn_eval = evaluators[weightSet];

        const int numJobs = problem.numJobs;
        const int numMachines = problem.numMachines;

        // Debug: Szczegóły problemu
        if (weightSet == 0 && problemIdx == 0) {
            //PrintProblemDetails(problem);
        }

        int base = (weightSet * numProblems + problemIdx) * maxOpsPerProblem;
        GPUOperation* local_ops = &ops_working[base];

        int jobScheduledOps[MAX_JOBS] = {0};
        int machine_times[MAX_MACHINES] = {0};
        
        int jobTypeCount[MAX_OP_TYPES] = {0};
        int opTypeCount[MAX_OP_TYPES] = {0};
        int opTypePerJobCount[MAX_JOBS][MAX_OP_TYPES] = {0};

        // Debug: Inicjalizacja danych
        if (weightSet == 0 && problemIdx == 0) {
            printf("[KERNEL] Inicjalizacja danych dla problemIdx=%d\n", problemIdx);
        }

        for (int jobID = 0; jobID < numJobs; ++jobID) {
            const GPUJob& job = problem.jobs[jobID];
            jobTypeCount[job.type]++;
            for (int opID = 0; opID < job.operationCount; ++opID) {
                GPUOperation& op = local_ops[job.operationsOffset + opID];
                opTypePerJobCount[jobID][op.type]++;
                opTypeCount[op.type]++;
            }
        }

        int local_makespan = 0;

        bool scheduled_any;
        do {
            scheduled_any = false;
            float bestScoreValue = -FLT_MAX;
            int bestJobID = -1, bestOpID = -1, bestMachineID = -1;
            int bestStartTime = 0;

            for (int jobID = 0; jobID < numJobs; ++jobID) {
                if (jobScheduledOps[jobID] == problem.jobs[jobID].operationCount)
                    continue;

                GPUJob& job = problem.jobs[jobID];

                for (int operationID = 0; operationID < job.operationCount; ++operationID) {
                    GPUOperation& operation = local_ops[job.operationsOffset + operationID];
                    if (operation.predecessorCount != 0) continue;

                    for (int m = 0; m < operation.eligibleCount; m++) {
                        int machineID = problem.eligibleMachines[operation.eligibleMachinesOffset + m];
                        int start_time = max(machine_times[machineID], operation.lastPredecessorEndTime);
                        int opMach_idx = operation.type * numMachines + machineID;
                        int pTime = problem.processingTimes[opMach_idx];

                        // Debug: Szczegóły operacji
                        if (weightSet == 0 && problemIdx == 0 && jobID == 0 && operationID == 0) {
                            //printf("[KERNEL] Operation details: jobID=%d, opID=%d, machineID=%d, start_time=%d, pTime=%d\n",
                                //   jobID, operationID, machineID, start_time, pTime);
                        }

                        float features[1 + 2 * MAX_MACHINES + 3 * MAX_OP_TYPES + 2 * MAX_JOB_TYPES] = {0.0f};

                        features[0] = static_cast<float>(start_time) - machine_times[machineID];

                        for (int i = 1; i < MAX_MACHINES + 1; ++i) {
                            features[i] = static_cast<float>(local_makespan - machine_times[i - 1]);
                        }
                        features[1 + machineID] = static_cast<float>(local_makespan - (start_time + pTime));
                        features[1 + MAX_MACHINES + machineID] = 1.0f;
                        features[1 + 2 * MAX_MACHINES + operation.type] = 1.0f;
                        
                        

                        // Zamień istniejący print features
                        if (weightSet == 0 && problemIdx == 0 && jobID == 0 && operationID == 0) {
                            printf("[DEBUG] Features (pierwsze 10): ");
                            for (int i = 0; i < min(10, 1 + 2 * MAX_MACHINES + 3 * MAX_OP_TYPES + 2 * MAX_JOB_TYPES); i++) {
                                printf("%.2f ", features[i]);
                            }
                            printf("...\n");
                            
                            // Wyświetl istotne sekcje features
                            printf("[DEBUG] Feature[0] (czas startu): %.2f\n", features[0]);
                            
                            printf("[DEBUG] Features[1-%d] (czasy maszyn): ", MAX_MACHINES);
                            for (int i = 1; i <= min(5, MAX_MACHINES); i++) {
                                printf("%.2f ", features[i]);
                            }
                            printf("...\n");
                            
                            // Sprawdź skrajne wartości
                            float min_val = FLT_MAX;
                            float max_val = -FLT_MAX;
                            int min_idx = -1, max_idx = -1;
                            for (int i = 0; i < (1 + 2 * MAX_MACHINES + 3 * MAX_OP_TYPES + 2 * MAX_JOB_TYPES); i++) {
                                if (features[i] < min_val) {
                                    min_val = features[i];
                                    min_idx = i;
                                }
                                if (features[i] > max_val) {
                                    max_val = features[i];
                                    max_idx = i;
                                }
                            }
                            printf("[DEBUG] Min/max features: min=%.2f (idx=%d), max=%.2f (idx=%d)\n", 
                                min_val, min_idx, max_val, max_idx);
}
                        

                        float score = nn_eval.Evaluate(features);//! Error: evaluate returns 0 or nans

                        // Debug: Wynik oceny
                        if (weightSet == 0 && problemIdx == 0 && jobID == 0 && operationID == 0) {
                            printf("[KERNEL] Score=%.2f\n", score);
                        }

                        if (score > bestScoreValue) {
                            bestScoreValue = score;
                            bestJobID = jobID;
                            bestOpID = operationID;
                            bestMachineID = machineID;
                            bestStartTime = start_time;
                        }
                    }
                }
            }

            if (bestJobID == -1) break;

            GPUJob& bestJob = problem.jobs[bestJobID];
            GPUOperation& bestOperation = local_ops[bestJob.operationsOffset + bestOpID];
            int opMach_idx = bestOperation.type * numMachines + bestMachineID;
            int pTime = problem.processingTimes[opMach_idx];

            int endTime = bestStartTime + pTime;

            jobScheduledOps[bestJobID]++;
            opTypePerJobCount[bestJobID][bestOperation.type]--;
            opTypeCount[bestOperation.type]--;

            if (jobScheduledOps[bestJobID] == bestJob.operationCount) {
                jobTypeCount[bestJob.type]--;
            }

            bestOperation.predecessorCount = -1;

            for (int s = 0; s < bestOperation.successorCount; ++s) {
                int successorID = problem.successorsIDs[bestOperation.successorsOffset + s];
                GPUOperation& successorOperation = local_ops[bestJob.operationsOffset + successorID];
                successorOperation.predecessorCount -= 1;
                successorOperation.lastPredecessorEndTime =
                    max(successorOperation.lastPredecessorEndTime, endTime);
            }

            machine_times[bestMachineID] = endTime;
            if (endTime > local_makespan) local_makespan = endTime;

            scheduled_any = true;
        } while (scheduled_any);

        makespan = static_cast<float>(local_makespan);
        shared_makespans[problemIdx] = makespan;

    } else {
        shared_makespans[problemIdx] = 0.0f;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < numProblems; ++i)
            sum += shared_makespans[i];
        results[weightSet] = sum / numProblems;
        printf("[KERNEL] weightSet=%d, avg makespan=%.2f\n", weightSet, results[weightSet]);
    }

    if (weightSet == 0 && problemIdx == 0) {
        printf("[KERNEL] SolveManyWeightsKernel end\n");
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