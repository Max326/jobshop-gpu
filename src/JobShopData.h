#ifndef JOB_SHOP_DATA_H
#define JOB_SHOP_DATA_H

#pragma once
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_set>
#include <vector>

#include "FileManager.h"

struct Operation {
	int type;
	std::vector<int> eligibleMachines;	// List of machines that can perform this operation
};

struct Job {
	int id;
	std::vector<Operation> operations;	// Ordered list of operation IDs
	int nextOpIndex = 0;				// Tracks which operation to schedule next
	int lastOpEndTime = 0;				// When the previous operation finished
};

struct Machine {
	int id;
};

class JobShopData
{
public:
	using json = nlohmann::json;

	int numMachines;
	int numJobs;
	int numOpTypes;	 // number of operation types
	std::vector<Job> jobs;
	std::vector<std::vector<int>> processingTimes;	// Macierz czasu przetwarzania [operacja][maszyna]

	// TODO: implement
	bool operInJobMultiplication = false;

	std::pair<int, int> jobCountRange;	  // Range of jobs
	std::pair<int, int> opDurationRange;  // Operation processing time range
	std::pair<int, int> opCountPerJobRange;
	std::pair<int, int> opFlexibilityRange;	 // Range of eligible machines for each opearation

	void SaveToJson(const std::string& filename) const {
		FileManager::EnsureDataDirExists();
		std::string full_path = FileManager::GetFullPath(filename);

		json j;
		j["numMachines"] = numMachines;
		j["numJobs"] = numJobs;
		j["numOpTypes"] = numOpTypes;

		j["jobs"] = json::array();
		for(const auto& job: jobs) {
			j["jobs"].push_back({{"id", job.id}, {"operations", job.operations}});
		}

		j["processingTimes"] = processingTimes;

		std::ofstream out(full_path);
		if(!out) {
			throw std::runtime_error("Failed to open file for writing: " + full_path);
		}
		out << j.dump(4);
		std::cout << "Data saved to: " << std::filesystem::absolute(full_path) << std::endl;
	}

	void LoadFromJson(const std::string& filename) {
		std::string full_path = FileManager::GetFullPath(filename);

		std::ifstream in(full_path);
		if(!in) {
			throw std::runtime_error("Failed to open file: " + full_path);
		}

		json j;
		in >> j;

		numMachines = j["numMachines"];
		numJobs = j["numJobs"];
		numOpTypes = j["numOpTypes"];

		jobs.clear();
		for(const auto& item: j["jobs"]) {
			jobs.push_back({item["id"].get<int>(),
							item["operations"].get<std::vector<int>>()});
		}

		processingTimes = j["processingTimes"].get<std::vector<std::vector<int>>>();

		if(!Validate()) {
			throw std::runtime_error("Loaded data failed validation");
		}
	}

	bool Validate() const {
		if(numMachines <= 0 || numJobs <= 0 || numOpTypes <= 0) return false;
		if(processingTimes.size() != numOpTypes) return false;
		for(const auto& row: processingTimes) {
			if(row.size() != numMachines) return false;
		}
		return true;
	}
};

inline JobShopData GenerateData() {
	JobShopData data;
	data.numMachines = 10;
	data.numJobs = 10;

	data.opCountPerJobRange = {5, 10};	// 5 to 10 operations per job
	data.opFlexibilityRange = {1, 3};	// 1 to 3 eligible machines per operation
	data.opDurationRange = {1, 10};		// Processing time from 1 to 10
	data.numOpTypes = data.opCountPerJobRange.second;

	std::srand(std::time(0));

	const int operationRangeRemainder = data.opCountPerJobRange.second - data.opCountPerJobRange.first + 1;
	const int flexibilityRangeRemainder = data.opFlexibilityRange.second - data.opFlexibilityRange.first + 1;
	const int durationRangeRemainder = data.opDurationRange.second - data.opDurationRange.first + 1;

	// 1. Generate processing times only for eligible machines
	data.processingTimes.resize(data.numOpTypes);

	data.processingTimes.resize(data.numOpTypes);
	for(int o = 0; o < data.numOpTypes; ++o) {
		// Create shuffled machine list for this operation type
		std::vector<int> machines(data.numMachines);
		std::iota(machines.begin(), machines.end(), 0);
		std::random_shuffle(machines.begin(), machines.end());

		// Select 1-3 eligible machines with random times
		int numEligible = data.opFlexibilityRange.first + rand() % flexibilityRangeRemainder;

		data.processingTimes[o].resize(data.numMachines, 0);  // Initialize all to 0 (ineligible)

		for(int m = 0; m < numEligible; ++m) {
			data.processingTimes[o][m] = data.opDurationRange.first + rand() % durationRangeRemainder;
		}
	}

	// Inicjalizacja jobów
	for(int j = 0; j < data.numJobs; ++j) {
		Job job;
		job.id = j;

		// Create shuffled operation types
        std::vector<int> opTypes(data.numOpTypes);
        std::iota(opTypes.begin(), opTypes.end(), 0);
        std::random_shuffle(opTypes.begin(), opTypes.end());

		
		for(int o = 0; o < data.opCountPerJobRange.first + rand() % operationRangeRemainder; ++o) {
			Operation op;
			op.type = (rand() % data.numOpTypes);  // Random operation type
			int numEligibleMachines = data.opFlexibilityRange.first + rand() % flexibilityRangeRemainder;

			for(int m = 0; m < numEligibleMachines; ++m) {
				std::unordered_set<int> selectedMachines;  // Track selected machines
				int machineId;
				do {
					machineId = rand() % data.numMachines;	// Randomly select a machine
				} while(selectedMachines.find(machineId) != selectedMachines.end());  // Ensure it's not already selected

				selectedMachines.insert(machineId);		   // Mark this machine as selected
				op.eligibleMachines.push_back(machineId);  // Assign the machine
			}

			job.operations.push_back(op);
		}
		data.jobs.push_back(job);

		// TODO: fix eligible machines for each operation
		// TODO: add rules to solve function
		// TODO: fix data loading
	}

	return data;
}

#endif	// JOB_SHOP_DATA_H