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

// Launch GPU kernel for batch solving
void JobShopHeuristic::SolveBatch(
    const GPUProblem* problems,
    SolutionManager::GPUSolutions* solutions,
    int numProblems) {
    auto eval = neuralNetwork.GetDeviceEvaluator();
    int threads = 1;
    int blocks = numProblems;
    
    cudaDeviceSetLimit(cudaLimitStackSize, 4096);  // Before kernel launch

    SolveFJSSPKernel<<<blocks, threads>>>(problems, eval, solutions, numProblems);
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

// Feature extraction for neural network (obsolete)
std::vector<float> JobShopHeuristic::ExtractFeatures(const JobShopData& data,
                                                     const Job& job,
                                                     const int& operationType,
                                                     const int& machineId,
                                                     const int& startTime,
                                                     const int& machineAvailableTime) const {
    std::vector<float> features;
    int waitTime = startTime - machineAvailableTime;
    int envelope = job.lastOpEndTime - machineAvailableTime;

    features.push_back(static_cast<float>(data.processingTimes[operationType][machineId]));
    features.push_back(static_cast<float>(waitTime));
    features.push_back(static_cast<float>(envelope));
    features.push_back(static_cast<float>(data.jobs[job.id].operations.size()));
    return features;
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

// GPU kernel for solving batch of problems
__global__ void SolveFJSSPKernel(
    const GPUProblem* problems,
    const NeuralNetwork::DeviceEvaluator nn_eval,
    SolutionManager::GPUSolutions* solutions,
    int total_problems) {

    int problem_id = blockIdx.x;
    if(problem_id >= total_problems) return;

    const GPUProblem problem = problems[problem_id];

    int* my_counts = solutions->allScheduleCounts +
                     problem_id * solutions->numMachines;
    OperationSchedule* my_schedule = solutions->allSchedules +
                                     problem_id * solutions->numMachines * solutions->maxOps;
    int* my_makespan = solutions->allMakespans + problem_id;

    int scheduledOps = 0;
    int machine_times[MAX_MACHINES] = {0};

    while(true) {
        float bestScoreValue = -FLT_MAX;
        int bestJobID = -1, bestOpID = -1, bestMachineID = -1;
        int bestStartTime = 0;

        // Iterate over all available operations in all jobs
        for(int jobID = 0; jobID < problem.numJobs; ++jobID) {
            const GPUJob& job = problem.jobs[jobID];
            for(int operationID = 0; operationID < job.operationCount; ++operationID) {
                GPUOperation& operation = problem.operations[job.operationsOffset + operationID];
                if (operation.predecessorCount != 0) continue;
        
                for(int m = 0; m < operation.eligibleCount; m++) {
                    int machineID = problem.eligibleMachines[operation.eligibleMachinesOffset + m];
                    int start_time = max(machine_times[machineID], operation.lastPredecessorEndTime);
                    int opMach_idx = operation.type * problem.numMachines + machineID;
                    int pTime = problem.processingTimes[opMach_idx];

                    float features[4] = {
                        static_cast<float>(pTime),
                        static_cast<float>(start_time - machine_times[machineID]),
                        static_cast<float>(4.0),
                        static_cast<float>(problem.jobs[jobID].operationCount)};
    
                    float score = nn_eval.Evaluate(features);
    
                    if(score > bestScoreValue) {
                        bestScoreValue = score;
                        bestJobID = jobID;
                        bestOpID = operationID;
                        bestMachineID = machineID;
                        bestStartTime= start_time;
                    }
                }
            }
        }

        if(bestJobID == -1) break;

        GPUJob& bestJob = problem.jobs[bestJobID];
        GPUOperation& bestOperation = problem.operations[bestJob.operationsOffset + bestOpID];
        int opMach_idx = bestOperation.type * problem.numMachines + bestMachineID;
        int pTime = problem.processingTimes[opMach_idx];

        int endTime = bestStartTime + pTime;

        bestOperation.predecessorCount = -1;

        for (int s = 0; s < bestOperation.successorCount; ++s) {
            int successorID = problem.successorsIDs[bestOperation.successorsOffset + s];
            GPUOperation& successorOperation = problem.operations[bestJob.operationsOffset + successorID];
            atomicSub(&successorOperation.predecessorCount, 1);
            successorOperation.lastPredecessorEndTime = 
                max(successorOperation.lastPredecessorEndTime, endTime);
        }

        int op_index = my_counts[bestMachineID]++;
        if(op_index < solutions->maxOps) {
            int flat_index = bestMachineID * solutions->maxOps + op_index;
            my_schedule[flat_index] = {
                bestJobID,
                bestOperation.type,
                bestStartTime,
                endTime};
        }

        machine_times[bestMachineID] = endTime;
        if(endTime > *my_makespan) *my_makespan = endTime;

        scheduledOps++;
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
        printf("Job %d (%d ops):\n", job.id, job.operationCount);

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