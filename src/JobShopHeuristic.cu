#include <algorithm>
#include <cfloat>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "JobShopHeuristic.cuh"

using json = nlohmann::json;

JobShopHeuristic::JobShopHeuristic(const std::vector<int>& topology)
	: neuralNetwork(topology) {}  // bezpośrednia inicjalizacja członka

JobShopHeuristic::JobShopHeuristic(const std::string& filename)
	: neuralNetwork(InitializeNetworkFromFile(filename)) {}

JobShopHeuristic::JobShopHeuristic(NeuralNetwork&& net)
	: neuralNetwork(std::move(net)) {}

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

		std::cout << "second weight: " << weights[1][2] << std::endl;
		std::cout << "second bias: " << biases[1][0] << std::endl;

		if(loaded_topology.empty() || weights.empty() || biases.empty()) {
			throw std::runtime_error("Invalid network data in file");
		}

		return NeuralNetwork(loaded_topology, &weights, &biases);
	} catch(const std::exception& e) {
		throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
	}
}

void JobShopHeuristic::CPUSolution::FromGPU(const SolutionManager::GPUSolution& gpuSol) {
	// 1. Download makespan
	cudaMemcpy(&makespan, gpuSol.makespan, sizeof(int), cudaMemcpyDeviceToHost);

	// 2. Resize schedule structure
	schedule.resize(gpuSol.numMachines);

	// 3. Download operation counts per machine
	std::vector<int> counts(gpuSol.numMachines);
	cudaMemcpy(counts.data(), gpuSol.scheduleCounts,
			   sizeof(int) * gpuSol.numMachines,
			   cudaMemcpyDeviceToHost);

	// 4. Download all operation entries (flat layout: [machine * MAX_OPS + op])
	std::vector<OperationSchedule> allOps(gpuSol.numMachines * MAX_OPS);
	cudaMemcpy(allOps.data(), gpuSol.schedule,
			   sizeof(OperationSchedule) * allOps.size(),
			   cudaMemcpyDeviceToHost);

	// 5. Reconstruct the 2D schedule
	for(int m = 0; m < gpuSol.numMachines; ++m) {
		schedule[m].clear();
		for(int i = 0; i < counts[m]; ++i) {
			schedule[m].push_back(allOps[m * MAX_OPS + i]);
		}
	}

	// Optional: validate/compute fallback makespan
	for(int m = 0; m < gpuSol.numMachines; ++m) {
		for(const auto& op: schedule[m]) {
			makespan = std::max(makespan, op.endTime);
		}
	}

	// Debug print
	int total_ops = 0;
	for(int m = 0; m < gpuSol.numMachines; ++m) {
		total_ops += counts[m];
	}
	std::cout << "Downloaded schedule with " << gpuSol.numMachines
			  << " machines, total ops: " << total_ops
			  << ", makespan: " << makespan << "\n";
}

SolutionManager::GPUSolution JobShopHeuristic::CPUSolution::ToGPU() const {
	SolutionManager::GPUSolution gpuSol;
	gpuSol.numMachines = schedule.size();

	// Allocate device memory
	cudaMalloc(&gpuSol.schedule, sizeof(OperationSchedule) * schedule.size() * MAX_OPS);
	cudaMalloc(&gpuSol.scheduleCounts, sizeof(int) * schedule.size());
	cudaMalloc(&gpuSol.makespan, sizeof(int));

	// Copy makespan
	cudaMemcpy(gpuSol.makespan, &makespan, sizeof(int), cudaMemcpyHostToDevice);

	// Flatten and copy schedule
	std::vector<OperationSchedule> flat_schedule;
	std::vector<int> counts;
	for(const auto& machine: schedule) {
		flat_schedule.insert(flat_schedule.end(), machine.begin(), machine.end());
		counts.push_back(machine.size());
	}

	cudaMemcpy(gpuSol.schedule, flat_schedule.data(),
			   sizeof(OperationSchedule) * flat_schedule.size(),
			   cudaMemcpyHostToDevice);
	cudaMemcpy(gpuSol.scheduleCounts, counts.data(),
			   sizeof(int) * counts.size(),
			   cudaMemcpyHostToDevice);

	return gpuSol;
}

void JobShopHeuristic::SolveBatch(
	const GPUProblem* problems,
	SolutionManager::GPUSolution* solutions,
	int numProblems) {
	auto eval = neuralNetwork.GetDeviceEvaluator();
	int threads = 256;
	int blocks = (numProblems + threads - 1) / threads;

	SolveFJSSPKernel<<<blocks, threads>>>(problems, eval, solutions, numProblems);
	cudaDeviceSynchronize();
}

SolutionManager::GPUSolution SolutionManager::CreateGPUSolution(int numMachines, int maxOps) {
	SolutionManager::GPUSolution sol;
	sol.numMachines = numMachines;
	cudaMalloc(&sol.schedule, sizeof(OperationSchedule) * numMachines * maxOps);
	cudaMalloc(&sol.scheduleCounts, sizeof(int) * numMachines);
	cudaMalloc(&sol.makespan, sizeof(int));	 // Allocate makespan on device
	cudaMemset(sol.scheduleCounts, 0, sizeof(int) * numMachines);
	cudaMemset(sol.makespan, 0, sizeof(int));  // Initialize makespan to 0
	return sol;
}

void SolutionManager::FreeGPUSolution(SolutionManager::GPUSolution& sol) {
	cudaFree(sol.schedule);
	cudaFree(sol.scheduleCounts);
	cudaFree(sol.makespan);	 // Free makespan memory
	sol = GPUSolution {};	 // Reset the struct
}

// JobShopHeuristic::CPUSolution JobShopHeuristic::Solve(const JobShopData& data) {  //! obsolete
// 	CPUSolution solution;
// 	solution.makespan = 0;
// 	solution.schedule.resize(data.numMachines);

// 	// solution.machineEndTimes.resize(data.numMachines, 0);

// 	JobShopData modifiedData = data;

// 	while(true) {
// 		// Znajdź dostępne operacje
// 		float bestScore = -FLT_MAX;
// 		int bestJobId = -1, bestOperationIdx = -1, bestMachineId = -1;

// 		int bestStartTime = 0;
// 		int bestProcessingTime = 0;

// 		for(int jobId = 0; jobId < modifiedData.numJobs; ++jobId) {
// 			auto& job = modifiedData.jobs[jobId];

// 			// Skip if no operations left or next operation isn't ready
// 			if(job.nextOpIndex >= job.operations.size()) continue;

// 			const auto& operation = job.operations[job.nextOpIndex];
// 			const int opType = operation.type;

// 			for(int machineId: operation.eligibleMachines) {
// 				if(modifiedData.processingTimes[opType][machineId] == 0) continue;

// 				// Get the time at which the machine will become available
// 				const int machineAvailableTime = solution.schedule[machineId].empty()
// 													 ? 0
// 													 : solution.schedule[machineId].back().endTime;

// 				const int startTime = std::max(machineAvailableTime, job.lastOpEndTime);

// 				// const int envelope = job.lastOpEndTime - machineAvailableTime;

// 				const int processingTime = data.processingTimes[opType][machineId];

// 				// TODO: rule 1: check machine availability and save that to "envelope"
// 				// TODO: rule 2: check for 'holes' in the schedule

// 				// TODO: implement dynamic NN input size, not from topology
// 				std::vector<float> features = ExtractFeatures(modifiedData, job, opType, machineId, startTime, machineAvailableTime);

// 				// std::cout << "Features: " << features[0] << ", " << features[1] << ", " << features[2] << std::endl;

// 				// features.resize(4);

// 				std::vector<float> output = neuralNetwork.Forward(features);

// 				float score = output[0];

// 				if(score > bestScore) {
// 					bestScore = score;
// 					bestJobId = jobId;
// 					bestOperationIdx = job.nextOpIndex;
// 					bestMachineId = machineId;
// 					bestStartTime = startTime;
// 					bestProcessingTime = processingTime;
// 				}
// 			}
// 		}

// 		if(bestJobId == -1) break;	// Wszystkie operacje zaplanowane

// 		// Zaplanuj operację na maszynie
// 		UpdateSchedule(modifiedData, bestJobId, bestOperationIdx, bestMachineId, solution);
// 	}

// 	return solution;
// }

std::vector<float> JobShopHeuristic::ExtractFeatures(const JobShopData& data,  //! obsolete
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

	features.push_back(static_cast<float>(envelope));  // TODO: vector?

	features.push_back(static_cast<float>(data.jobs[job.id].operations.size()));

	return features;
}

void JobShopHeuristic::UpdateSchedule(JobShopData& data, int jobId, int operationIdx,  //! obsolete
									  int machineId, CPUSolution& solution) {
	auto& job = data.jobs[jobId];
	const auto& operation = job.operations[operationIdx];

	int processingTime = data.processingTimes[operation.type][machineId];

	// Validate processing time
	if(processingTime <= 0) {
		std::cerr << "Warning: Attempted to schedule zero-duration operation\n";
		return;
	}

	// Get machine's last operation end time (0 if no operations yet)
	int machineAvailableTime = solution.schedule[machineId].empty()
								   ? 0
								   : solution.schedule[machineId].back().endTime;

	int startTime = std::max(machineAvailableTime, job.lastOpEndTime);	// TODO: source the last operation end time from the schedule

	int endTime = startTime + processingTime;

	solution.schedule[machineId].push_back({jobId, operation.type, startTime, endTime});

	job.lastOpEndTime = endTime;  // Update job's last operation end time
	job.nextOpIndex++;

	solution.makespan = std::max(solution.makespan, endTime);

	// data.jobs[jobId].operations.pop_back();	 // Remove the scheduled operation
}

void JobShopHeuristic::PrintSchedule(const CPUSolution& solution, const JobShopData& data) {
	std::cout << "\n=== FINAL SCHEDULE ===" << std::endl;

	for(int machineId = 0; machineId < solution.schedule.size(); ++machineId) {
		std::cout << "M" << machineId << ": [";

		int currentTime = 0;
		bool firstElement = true;
		const auto& machineSchedule = solution.schedule[machineId];

		if(machineSchedule.empty()) {
			std::cout << "idle";
		} else {
			for(const auto& scheduledOp: machineSchedule) {
				// Add waiting time before operation
				if(scheduledOp.startTime > currentTime) {
					if(!firstElement) std::cout << "][";
					std::cout << "w-" << (scheduledOp.startTime - currentTime);
					firstElement = false;
					currentTime = scheduledOp.startTime;
				}

				// Add operation with job ID
				if(!firstElement) std::cout << "][";
				std::cout << "t=" << scheduledOp.startTime << ",j" << scheduledOp.jobId << "-o" << scheduledOp.opType
						  << "-" << (scheduledOp.endTime - scheduledOp.startTime);
				currentTime = scheduledOp.endTime;
				firstElement = false;
			}
		}

		std::cout << "]" << std::endl;
	}

	std::cout << "Makespan: " << solution.makespan << "\n"
			  << std::endl;
}

__global__ void SolveFJSSPKernel(
	const GPUProblem* problems,
	const NeuralNetwork::DeviceEvaluator nn_eval,
	SolutionManager::GPUSolution* solutions,
	int total_problems) {
	int problem_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(problem_id >= total_problems) return;

	GPUProblem problem = problems[problem_id];
	SolutionManager::GPUSolution solution = solutions[problem_id];

	// solution.makespan = 0;

	// Validation checks
	if(problem.numJobs <= 0 || problem.numMachines <= 0) return;
	if(problem.jobs == nullptr || problem.processingTimes == nullptr) return;

	int machine_times[MAX_MACHINES] = {0};
	int job_next[MAX_JOBS] = {0};
	int job_last[MAX_JOBS] = {0};

	while(true) {
		float best_score = -FLT_MAX;
		int best_job = -1, best_op = -1, best_machine = -1;
		int best_start_time = 0;

		for(int job = 0; job < problem.numJobs; job++) {
			// Skip if no operations left
			if(job_next[job] >= problem.jobs[job].operationCount) continue;

			GPUOperation op = problem.jobs[job].operations[job_next[job]];

			// Validate operation
			if(op.eligibleCount <= 0 || op.eligibleMachines == nullptr) continue;

			for(int m = 0; m < op.eligibleCount; m++) {
				int machine = op.eligibleMachines[m];

				// Validate machine index
				if(machine < 0 || machine >= problem.numMachines) continue;

				int start_time = max(machine_times[machine], job_last[job]);

				// Calculate processing time with bounds checking
				int ptime = 0;
				int type_idx = op.type * problem.numMachines + machine;
				if(type_idx >= 0 && type_idx < problem.numOpTypes * problem.numMachines) {
					ptime = problem.processingTimes[type_idx];
				}
				if(ptime <= 0) continue;

				float features[4] = {
					static_cast<float>(ptime),
					static_cast<float>(start_time - machine_times[machine]),
					static_cast<float>(job_last[job] - machine_times[machine]),
					static_cast<float>(problem.jobs[job].operationCount)};

				float score = nn_eval.Evaluate(features);

				if(score > best_score) {
					best_score = score;
					best_job = job;
					best_op = job_next[job];
					best_machine = machine;
					best_start_time = start_time;
				}

				atomicMax(solution.makespan, start_time + ptime);
			}
		}

		if(best_job == -1) break;  // No more operations to schedule

		// Validate indices before scheduling
		if(best_machine < 0 || best_machine >= problem.numMachines) continue;
		if(best_job < 0 || best_job >= problem.numJobs) continue;

		// Calculate processing time with validation
		int ptime = 0;
		GPUOperation best_op_data = problem.jobs[best_job].operations[best_op];
		int type_idx = best_op_data.type * problem.numMachines + best_machine;
		if(type_idx >= 0 && type_idx < problem.numOpTypes * problem.numMachines) {
			ptime = problem.processingTimes[type_idx];
		}
		if(ptime <= 0) continue;

		int end_time = best_start_time + ptime;

		// Schedule the operation
		int op_index = atomicAdd(&solution.scheduleCounts[best_machine], 1);
		if(op_index < MAX_OPS) {  // Prevent buffer overflow
			solution.schedule[best_machine * MAX_OPS + op_index] = {
				best_job,
				best_op_data.type,
				end_time - ptime,
				end_time};
		}

		machine_times[best_machine] = end_time;
		job_last[best_job] = end_time;
		job_next[best_job]++;

		// printf("Scheduled: Job %d OpType %d on Machine %d (%d-%d), makespan: %d\n",
		// 	   best_job, best_op_data.type, best_machine,
		// 	   end_time - ptime, end_time, solution.makespan);
	}
}