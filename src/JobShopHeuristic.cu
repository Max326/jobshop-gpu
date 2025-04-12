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

void JobShopHeuristic::CPUSolution::FromGPU(const SolutionManager::GPUSolutions& gpuSols, int problemId) {
	// Calculate offsets
	int counts_offset = problemId * gpuSols.numMachines;
	int schedule_offset = problemId * gpuSols.numMachines * gpuSols.maxOps;

	// Download counts
	std::vector<int> counts(gpuSols.numMachines);
	cudaMemcpy(counts.data(), gpuSols.allScheduleCounts + counts_offset,
			   sizeof(int) * gpuSols.numMachines, cudaMemcpyDeviceToHost);

	// Download makespan
	cudaMemcpy(&makespan, gpuSols.allMakespans + problemId,
			   sizeof(int), cudaMemcpyDeviceToHost);

	// Download schedule
	std::vector<OperationSchedule> allOps(gpuSols.numMachines * gpuSols.maxOps);
	cudaMemcpy(allOps.data(), gpuSols.allSchedules + schedule_offset,
			   sizeof(OperationSchedule) * allOps.size(), cudaMemcpyDeviceToHost);

	// 4. Reconstruct 2D schedule
	schedule.resize(gpuSols.numMachines);
	for(int m = 0; m < gpuSols.numMachines; ++m) {
		schedule[m].clear();
		for(int i = 0; i < counts[m] && i < gpuSols.maxOps; ++i) {
			int idx = m * gpuSols.maxOps + i;
			schedule[m].push_back(allOps[idx]);
		}
	}

	// Debug print
	int total_ops = 0;
	for(int m = 0; m < gpuSols.numMachines; ++m) {
		total_ops += counts[m];
	}
	// std::cout << "Downloaded schedule with " << gpuSols.numMachines
	// 		  << " machines, total ops: " << total_ops
	// 		  << ", makespan: " << makespan << "\n";
}

SolutionManager::GPUSolutions JobShopHeuristic::CPUSolution::ToGPU() const {
	SolutionManager::GPUSolutions gpuSol;
	gpuSol.numMachines = schedule.size();

	// Allocate device memory
	cudaMalloc(&gpuSol.allSchedules, sizeof(OperationSchedule) * schedule.size() * MAX_OPS);
	cudaMalloc(&gpuSol.allScheduleCounts, sizeof(int) * schedule.size());
	cudaMalloc(&gpuSol.allMakespans, sizeof(int));

	// Copy makespan
	cudaMemcpy(gpuSol.allMakespans, &makespan, sizeof(int), cudaMemcpyHostToDevice);

	// Flatten and copy schedule
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

void JobShopHeuristic::SolveBatch(
	const GPUProblem* problems,
	SolutionManager::GPUSolutions* solutions,
	int numProblems) {
	auto eval = neuralNetwork.GetDeviceEvaluator();
	int threads = 256;
	int blocks = (numProblems + threads - 1) / threads;

	SolveFJSSPKernel<<<blocks, threads>>>(problems, eval, solutions, numProblems);
	cudaDeviceSynchronize();
}

SolutionManager::GPUSolutions SolutionManager::CreateGPUSolutions(int numProblems, int numMachines, int maxOps) {
	GPUSolutions solutions;
	solutions.numProblems = numProblems;
	solutions.numMachines = numMachines;
	solutions.maxOps = maxOps;

	size_t schedule_size = sizeof(OperationSchedule) * numMachines * maxOps * numProblems;
	cudaMalloc(&solutions.allSchedules, schedule_size);
	cudaMemset(solutions.allSchedules, 0, schedule_size);  // Initialize to zero

	size_t counts_size = numProblems * numMachines * sizeof(int);
	cudaMalloc(&solutions.allScheduleCounts, counts_size);
	cudaMemset(solutions.allScheduleCounts, 0, counts_size);  // Initialize to zero

	cudaMalloc(&solutions.allMakespans, sizeof(int) * numProblems);
	// int zero = 0;
	// cudaMemcpy(solutions.allMakespans, &zero, sizeof(int) * numProblems, cudaMemcpyHostToDevice);
	cudaMemset(solutions.allMakespans, 0, numProblems * sizeof(int));

	return solutions;
}

void SolutionManager::FreeGPUSolutions(SolutionManager::GPUSolutions& sols) {
	cudaFree(sols.allSchedules);
	cudaFree(sols.allScheduleCounts);
	cudaFree(sols.allMakespans);  // Free makespan memory
	sols = GPUSolutions {};		  // Reset the struct
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
	SolutionManager::GPUSolutions* solutions,
	int total_problems) {
	int problem_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(problem_id >= total_problems) return;

	const GPUProblem problem = problems[problem_id];
	SolutionManager::GPUSolutions solution = solutions[problem_id];

	int* my_counts = solutions->allScheduleCounts +
					 problem_id * solutions->numMachines;

	OperationSchedule* my_schedule = solutions->allSchedules +
									 problem_id * solutions->numMachines * solutions->maxOps;

	int* my_makespan = solutions->allMakespans + problem_id;

	// printf("Solving problem %d\n", problem_id);
	int scheduledOps = 0;

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
		int op_index = atomicAdd(&my_counts[best_machine], 1);
		if(op_index < solutions->maxOps) {
			int flat_index = best_machine * solutions->maxOps + op_index;
			my_schedule[flat_index] = {
				best_job,
				best_op_data.type,
				best_start_time,
				end_time};
		}

		machine_times[best_machine] = end_time;
		job_last[best_job] = end_time;
		job_next[best_job]++;

		atomicMax(my_makespan, end_time);

		// printf("%d: Scheduled: Job %d, OpType %d on Machine %d (%d-%d), makespan: %d\n",
		// 	   scheduledOps, best_job, best_op_data.type, best_machine,
		// 	   end_time - ptime, end_time, *solution.makespan);

		scheduledOps++;
	}
}

/*
JobShopHeuristic::CPUSolution JobShopHeuristic::Solve(const JobShopData& data) {  //! obsolete
	CPUSolution solution;
	solution.makespan = 0;
	solution.schedule.resize(data.numMachines);

	// solution.machineEndTimes.resize(data.numMachines, 0);

	JobShopData modifiedData = data;

	while(true) {
		// Znajdź dostępne operacje
		float bestScore = -FLT_MAX;
		int bestJobId = -1, bestOperationIdx = -1, bestMachineId = -1;

		int bestStartTime = 0;
		int bestProcessingTime = 0;

		for(int jobId = 0; jobId < modifiedData.numJobs; ++jobId) {
			auto& job = modifiedData.jobs[jobId];

			// Skip if no operations left or next operation isn't ready
			if(job.nextOpIndex >= job.operations.size()) continue;

			const auto& operation = job.operations[job.nextOpIndex];
			const int opType = operation.type;

			for(int machineId: operation.eligibleMachines) {
				if(modifiedData.processingTimes[opType][machineId] == 0) continue;

				// Get the time at which the machine will become available
				const int machineAvailableTime = solution.schedule[machineId].empty()
													 ? 0
													 : solution.schedule[machineId].back().endTime;

				const int startTime = std::max(machineAvailableTime, job.lastOpEndTime);

				// const int envelope = job.lastOpEndTime - machineAvailableTime;

				const int processingTime = data.processingTimes[opType][machineId];

				// TODO: rule 1: check machine availability and save that to "envelope"
				// TODO: rule 2: check for 'holes' in the schedule

				// TODO: implement dynamic NN input size, not from topology
				std::vector<float> features = ExtractFeatures(modifiedData, job, opType, machineId, startTime, machineAvailableTime);

				// std::cout << "Features: " << features[0] << ", " << features[1] << ", " << features[2] << std::endl;

				// features.resize(4);

				std::vector<float> output = neuralNetwork.Forward(features);

				float score = output[0];

				if(score > bestScore) {
					bestScore = score;
					bestJobId = jobId;
					bestOperationIdx = job.nextOpIndex;
					bestMachineId = machineId;
					bestStartTime = startTime;
					bestProcessingTime = processingTime;
				}
			}
		}

		if(bestJobId == -1) break;	// Wszystkie operacje zaplanowane

		// Zaplanuj operację na maszynie
		UpdateSchedule(modifiedData, bestJobId, bestOperationIdx, bestMachineId, solution);
	}

	return solution;
}
*/

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