#include "JobShopHeuristic.h"

#include <algorithm>
#include <cfloat>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

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
		int bestJobId = -1, bestOperationId = -1, bestMachineId = -1;

		for(int jobId = 0; jobId < modifiedData.numJobs; ++jobId) {
			// if(modifiedData.jobs[jobId].operations.empty()) continue;
			auto& job = modifiedData.jobs[jobId];

			// Skip if no operations left or next operation isn't ready
			if(job.nextOpIndex >= job.operations.size()) continue;

			// int operationId = job.operations[job.nextOpIndex];	// Get NEXT operation
			const auto& operation = job.operations[job.nextOpIndex];

			// for (const auto& [machineId] : operation.eligibleMachines) {
			// }


			for(int machineId = 0; machineId < modifiedData.numMachines; ++machineId) {
				if(modifiedData.processingTimes[operationId][machineId] == 0) continue;	 // TODO: add more rules

				// Check machine availability
				int machineAvailableTime = solution.schedule[machineId].empty()
											   ? 0
											   : solution.schedule[machineId].back().endTime;

				// Operation can only start after previous operation in job completes
				int startTime = std::max(machineAvailableTime, job.lastOpEndTime);

				// TODO: rule 1: if machine is busy, skip
				// if(solution.machineEndTimes[machineId] > solution.jobEndTimes[jobId]) {
				// 	continue;  // Maszyna zajęta
				// }

				// TODO: rule 2: check for 'holes' in the schedule
				// TODO: rule 3: check if machine matches operation type
				// TODO: rule 4:

				// Przygotuj wektor cech
				int waitTime = machineAvailableTime - startTime;

				std::vector<float> features = ExtractFeatures(modifiedData, jobId, operationId, machineId, waitTime);

				// std::cout << "Features: " << features[0] << ", " << features[1] << ", " << features[2] << std::endl;

				features.resize(2);

				std::vector<float> output = neuralNetwork.Forward(features);

				float score = output[0];

				if(score > bestScore) {
					bestScore = score;
					bestJobId = jobId;
					bestOperationId = operationId;
					bestMachineId = machineId;
				}
			}
		}

		if(bestJobId == -1) break;	// Wszystkie operacje zaplanowane

		// Zaplanuj operację na maszynie
		UpdateSchedule(modifiedData, bestJobId, bestOperationId, bestMachineId, solution);
	}

	return solution;
}

std::vector<float> JobShopHeuristic::ExtractFeatures(const JobShopData& data, int jobId, int operationId, int machineId, int waitTime) {
	std::vector<float> features;

	features.push_back(static_cast<float>(data.processingTimes[operationId][machineId]));

	features.push_back(static_cast<float>(data.jobs[jobId].operations.size()));

	features.push_back(static_cast<float>(waitTime));

	return features;
}

void JobShopHeuristic::UpdateSchedule(JobShopData& data, int jobId, int operationId,
									  int machineId, Solution& solution) {
	auto& job = data.jobs[jobId];

	// Get machine's last operation end time (0 if no operations yet)
	int machineAvailableTime = solution.schedule[machineId].empty()
								   ? 0
								   : solution.schedule[machineId].back().endTime;

	// Start time is the later of machine or job availability
	// int startTime = std::max(solution.jobEndTimes[jobId],
	//  solution.machineEndTimes[machineId]);

	int startTime = std::max(machineAvailableTime, job.lastOpEndTime);	// TODO: source the last operation end time from the schedule

	int processingTime = data.processingTimes[operationId][machineId];
	int endTime = startTime + processingTime;

	solution.schedule[machineId].push_back({jobId, operationId, startTime, endTime});

	job.lastOpEndTime = endTime;  // Update job's last operation end time
	job.nextOpIndex++;

	solution.makespan = std::max(solution.makespan, endTime);

	// data.jobs[jobId].operations.pop_back();	 // Remove the scheduled operation
}

void JobShopHeuristic::PrintSchedule(const Solution& solution, const JobShopData& data) {
    std::cout << "\n=== FINAL SCHEDULE ===" << std::endl;

    for (int machineId = 0; machineId < solution.schedule.size(); ++machineId) {
        std::cout << "M" << machineId << ": [";

        int currentTime = 0;
        bool firstElement = true;
        const auto& machineSchedule = solution.schedule[machineId];

        if (machineSchedule.empty()) {
            std::cout << "idle";
        } else {
            for (const auto& scheduledOp : machineSchedule) {
                // Add waiting time before operation
                if (scheduledOp.startTime > currentTime) {
                    if (!firstElement) std::cout << "][";
                    std::cout << "w-" << (scheduledOp.startTime - currentTime);
                    firstElement = false;
                    currentTime = scheduledOp.startTime;
                }

                // Add operation with job ID
                if (!firstElement) std::cout << "][";
                std::cout << "j" << scheduledOp.jobId << "-o" << scheduledOp.opId 
                          << "-" << (scheduledOp.endTime - scheduledOp.startTime);
                currentTime = scheduledOp.endTime;
                firstElement = false;
            }
        }

        std::cout << "]" << std::endl;
    }

    std::cout << "Makespan: " << solution.makespan << "\n" << std::endl;
}