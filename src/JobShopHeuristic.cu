#include <cuda_runtime.h>

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

JobShopHeuristic::Solution JobShopHeuristic::Solve(const JobShopData& data) {
	Solution solution;
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

	features.push_back(static_cast<float>(envelope));  // TODO: vector?

	features.push_back(static_cast<float>(data.jobs[job.id].operations.size()));

	return features;
}

void JobShopHeuristic::UpdateSchedule(JobShopData& data, int jobId, int operationIdx,
									  int machineId, Solution& solution) {
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

	// Start time is the later of machine or job availability
	// int startTime = std::max(solution.jobEndTimes[jobId],
	//  solution.machineEndTimes[machineId]);

	int startTime = std::max(machineAvailableTime, job.lastOpEndTime);	// TODO: source the last operation end time from the schedule

	int endTime = startTime + processingTime;

	solution.schedule[machineId].push_back({jobId, operation.type, startTime, endTime});

	job.lastOpEndTime = endTime;  // Update job's last operation end time
	job.nextOpIndex++;

	solution.makespan = std::max(solution.makespan, endTime);

	// data.jobs[jobId].operations.pop_back();	 // Remove the scheduled operation
}

void JobShopHeuristic::PrintSchedule(const Solution& solution, const JobShopData& data) {
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
				std::cout << "t=" << scheduledOp.startTime << ",j" << scheduledOp.jobId << "-o" << scheduledOp.opId
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