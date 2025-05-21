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
	int numProblems_per_block,	  // Renamed for clarity: num FJSS problems this block will handle
	int numWeights_total_blocks,  // Renamed for clarity: total NNs, so total blocks
	int maxOpsPerProblem,
	cudaStream_t stream,				 // Removed default stream = 0 as it's passed from evaluator
	int nn_total_params_for_one_network	 // <<< NEW PARAMETER
) {
	int threads_per_block = 64;	 // This is your blockDim.x
	int total_cuda_blocks = numWeights_total_blocks;

	// Calculate dynamic shared memory size:
	// (threads_per_block * sizeof(float) for shared_makespans)
	// + (nn_total_params_for_one_network * sizeof(float) for combined weights & biases of ONE network)
	size_t dynamic_shared_mem_size = (threads_per_block * sizeof(float)) + (nn_total_params_for_one_network * sizeof(float));

	cudaDeviceSetLimit(cudaLimitStackSize, 4096);
	// int reset_value = 0; // If gpu_error_flag is used
	// cudaMemcpyToSymbol(gpu_error_flag, &reset_value, sizeof(int), 0, cudaMemcpyHostToDevice);

	SolveManyWeightsKernel<<<total_cuda_blocks, threads_per_block, dynamic_shared_mem_size, stream>>>(
		problems,
		evaluators,
		ops_working,
		results,
		numProblems_per_block,	// This is how many problems each block should iterate up to.
		maxOpsPerProblem);

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
	const NeuralNetwork::DeviceEvaluator* evaluators,  // This points to DeviceEvaluators in global memory
	GPUOperation* ops_working,
	float* results,
	int numProblemsToSolvePerBlock,	 // Renamed for clarity (was numProblems)
	int maxOpsPerProblem) {
	// if(gpu_error_flag) { return; } // Re-enable if needed, but you said removing checks helped

	// Combined dynamic shared memory
	extern __shared__ float shared_block_data[];

	// Partition 1: Makespans for each problem solved by threads in this block
	// blockDim.x is the number of threads in this block (e.g., 64)
	float* shared_makespans = shared_block_data;

	// --- Identify current weight set and problem for this thread ---
	int weightSet = blockIdx.x;			  // Each block handles one weightSet
	int problemIdxInBlock = threadIdx.x;  // Each thread in block handles one FJSS problem for this weightSet

	// --- Load NN Parameters into Shared Memory ---
	const NeuralNetwork::DeviceEvaluator& nn_eval_global_ptr = evaluators[weightSet];  // Get the evaluator for this block

	// Calculate total weights and biases for this NN
	// This must be done by all threads or broadcast, as it's needed for shared mem partitioning
	int nn_total_weights = 0;
	int nn_total_biases = 0;
	if(nn_eval_global_ptr.num_layers > 0) {
		for(int i = 1; i < nn_eval_global_ptr.num_layers; ++i) {
			nn_total_weights += nn_eval_global_ptr.d_topology[i - 1] * nn_eval_global_ptr.d_topology[i];
			nn_total_biases += nn_eval_global_ptr.d_topology[i];
		}
	}
	// else: handle error or assume valid if pre-checked

	// Partition 2: Storage for NN weights for this block (starts after shared_makespans)
	float* sm_weights = shared_block_data + blockDim.x;

	// Partition 3: Storage for NN biases for this block (starts after sm_weights)
	float* sm_biases = shared_block_data + blockDim.x + nn_total_weights;

	// Cooperatively load NN parameters from global to shared memory
	// nn_eval_global_ptr.weights and nn_eval_global_ptr.biases point to global device memory
	
	// Load weights cooperatively and coalesced
	// Calculate how many elements each thread might load in total passes
	int num_passes_weights = (nn_total_weights + blockDim.x - 1) / blockDim.x;
	for (int pass = 0; pass < num_passes_weights; ++pass) {
		int current_element_idx = pass * blockDim.x + threadIdx.x;
		if (current_element_idx < nn_total_weights) {
			sm_weights[current_element_idx] = nn_eval_global_ptr.weights[current_element_idx];
		}
	}

	// Load biases cooperatively and coalesced
	int num_passes_biases = (nn_total_biases + blockDim.x - 1) / blockDim.x;
	for (int pass = 0; pass < num_passes_biases; ++pass) {
		int current_element_idx = pass * blockDim.x + threadIdx.x;
		if (current_element_idx < nn_total_biases) {
			sm_biases[current_element_idx] = nn_eval_global_ptr.biases[current_element_idx];
		}
	}
	__syncthreads();  // IMPORTANT: Ensure all threads finish loading before any thread proceeds

	// --- Main problem-solving logic ---
	float makespan_val = 0.0f;	// Changed variable name to avoid conflict
	// numProblemsToSolvePerBlock is the number of FJSS problems this block is responsible for
	if(problemIdxInBlock < numProblemsToSolvePerBlock) {
		const GPUProblem problem = problems[problemIdxInBlock];	 // Assuming 'problems' array is correctly indexed for the batch

		// local_ops indexing seems correct from your previous structure
		int base_op_idx = (weightSet * numProblemsToSolvePerBlock + problemIdxInBlock) * maxOpsPerProblem;
		GPUOperation* local_ops = &ops_working[base_op_idx];

		// ... (rest of your existing problem setup: jobScheduledOps, machine_times, etc. from JobShopHeuristic.cu[6]) ...
		int jobScheduledOps[MAX_JOBS] = {0};
		int machine_times[MAX_MACHINES] = {0};

		int jobTypeCount[MAX_OP_TYPES] = {0};
		int opTypeCount[MAX_OP_TYPES] = {0};
		int opTypePerJobCount[MAX_JOBS][MAX_OP_TYPES] = {0};

		const int numJobs = problem.numJobs;
		const int numMachines = problem.numMachines;

		for(int jobID = 0; jobID < numJobs; ++jobID) {
			const GPUJob& job = problem.jobs[jobID];
			jobTypeCount[job.type]++;
			for(int opID = 0; opID < job.operationCount; ++opID) {
				GPUOperation& op = local_ops[job.operationsOffset + opID];
				opTypePerJobCount[jobID][op.type]++;
				opTypeCount[op.type]++;
			}
		}

		
		int current_local_makespan = 0;	 // Renamed to avoid conflict
		bool scheduled_any;
		do {
			scheduled_any = false;
			float bestScoreValue = -FLT_MAX;
			int bestJobID = -1, bestOpID = -1, bestMachineID = -1;
			int bestStartTime = 0;

			for(int jobID = 0; jobID < numJobs; ++jobID) {
				if(jobScheduledOps[jobID] == problem.jobs[jobID].operationCount)
					continue;

				GPUJob& job = problem.jobs[jobID];

				for(int operationID = 0; operationID < job.operationCount; ++operationID) {
					GPUOperation& operation = local_ops[job.operationsOffset + operationID];
					if(operation.predecessorCount != 0) continue;

					for(int m = 0; m < operation.eligibleCount; m++) {
						int machineID = problem.eligibleMachines[operation.eligibleMachinesOffset + m];
						int start_time = max(machine_times[machineID], operation.lastPredecessorEndTime);
						int opMach_idx = operation.type * numMachines + machineID;
						int pTime = problem.processingTimes[opMach_idx];

						// Debug: op details
						/*                         if (weightSet == 0 && problemIdx == 0 && jobID == 0 && operationID == 0) {
													printf("[KERNEL] Operation details: jobID=%d, opID=%d, machineID=%d, start_time=%d, pTime=%d\n",
														  jobID, operationID, machineID, start_time, pTime);
												}
						 */
						float features[1 + 2 * MAX_MACHINES + 3 * MAX_OP_TYPES + 2 * MAX_JOB_TYPES] = {0.0f};

						features[0] = static_cast<float>(start_time) - machine_times[machineID];  // wasted time

						for(int i = 1; i < MAX_MACHINES + 1; ++i) {
							features[i] = static_cast<float>(current_local_makespan - machine_times[i - 1]);  // envelope
						}
						features[1 + machineID] = static_cast<float>(current_local_makespan - (start_time + pTime));  // envelope for current machine

						features[1 + MAX_MACHINES + machineID] = 1.0f;			 // one hot machine encoding
						features[1 + 2 * MAX_MACHINES + operation.type] = 1.0f;	 // one hot operation type encoding

						const float SCALE_FACTOR = 100.0f;
						const float inv_SCALE_FACTOR = 1.0 / SCALE_FACTOR;
						// normalize nn inputs (it may like it better)
						features[0] *= inv_SCALE_FACTOR;
						for(int i = 1; i < MAX_MACHINES + 1; ++i) {
							features[i] *= inv_SCALE_FACTOR;
						}
						features[1 + machineID] *= inv_SCALE_FACTOR;

						float score = nn_eval_global_ptr.Evaluate(features, sm_weights, sm_biases);
						// float score2 = nn_eval_global_ptr.Evaluate(features); // For debug

						// if (score != score2) {
						// 	printf("[KERNEL] Score mismatch: %f vs %f\n", score, score2);
						// 	return;
						// }

						if(score > bestScoreValue) {
							bestScoreValue = score;
							bestJobID = jobID;
							bestOpID = operationID;
							bestMachineID = machineID;
							bestStartTime = start_time;
						}
					}
				}
			}

			if(bestJobID == -1) break;


			GPUJob& bestJob = problem.jobs[bestJobID];	// problem is const, so bestJob needs to be const if problem.jobs not modifiable
			GPUOperation& bestOperation = local_ops[bestJob.operationsOffset + bestOpID];
			int opMach_idx = bestOperation.type * problem.numMachines + bestMachineID;	// Use problem.numMachines
			int pTime = problem.processingTimes[opMach_idx];
			int endTime = bestStartTime + pTime;

			jobScheduledOps[bestJobID]++;
			opTypePerJobCount[bestJobID][bestOperation.type]--;
			opTypeCount[bestOperation.type]--;

			if(jobScheduledOps[bestJobID] == bestJob.operationCount) {
				jobTypeCount[bestJob.type]--;
			}

			bestOperation.predecessorCount = -1;
			for(int s = 0; s < bestOperation.successorCount; ++s) {
				int successor_op_array_idx = problem.successorsIDs[bestOperation.successorsOffset + s];
				// The successorID is an index relative to the start of the current job's operations.
				GPUOperation& successorOperation = local_ops[bestJob.operationsOffset + successor_op_array_idx];
				successorOperation.predecessorCount -= 1;
				successorOperation.lastPredecessorEndTime = max(successorOperation.lastPredecessorEndTime, endTime);
			}
			machine_times[bestMachineID] = endTime;
			if(endTime > current_local_makespan) current_local_makespan = endTime;
			scheduled_any = true;
		} while(scheduled_any);
		makespan_val = static_cast<float>(current_local_makespan);
		shared_makespans[problemIdxInBlock] = makespan_val;
	} else {
		// Threads outside the numProblemsToSolvePerBlock range (e.g. if blockDim.x > numProblemsToSolvePerBlock)
		shared_makespans[problemIdxInBlock] = 0.0f;
	}

	__syncthreads();

	// Reduction to calculate average makespan for this weightSet (block)
	if(threadIdx.x == 0) {
		float sum = 0.0f;
		for(int i = 0; i < numProblemsToSolvePerBlock; ++i) {  // Iterate up to actual problems solved
			sum += shared_makespans[i];
		}
		if(numProblemsToSolvePerBlock > 0) {
			results[weightSet] = sum / numProblemsToSolvePerBlock;
		} else {
			results[weightSet] = 0.0f;
		}
		//printf("[KERNEL] weightSet=%d, avg makespan=%.2f\n", weightSet, results[weightSet]);  // Keep for debug if needed
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