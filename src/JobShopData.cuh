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

/* CPU Implementation */

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

class JobShopData
{
public:
	using json = nlohmann::json;

	int numMachines;
	int numJobs;
	int numOpTypes;	 // number of operation types
	std::vector<Job> jobs;
	std::vector<std::vector<int>> processingTimes;	// processing times for each operation type on each machine

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
			json jobJson;
			jobJson["id"] = job.id;
			jobJson["nextOpIndex"] = job.nextOpIndex;
			jobJson["lastOpEndTime"] = job.lastOpEndTime;

			json operationsJson = json::array();
			for(const auto& op: job.operations) {
				json opJson;
				opJson["type"] = op.type;
				opJson["eligibleMachines"] = op.eligibleMachines;
				operationsJson.push_back(opJson);
			}
			jobJson["operations"] = operationsJson;

			j["jobs"].push_back(jobJson);
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
		try {
			in >> j;
		} catch(const json::parse_error& e) {
			throw std::runtime_error("JSON parse error: " + std::string(e.what()));
		}

		try {
			numMachines = j.at("numMachines").get<int>();
			numJobs = j.at("numJobs").get<int>();
			numOpTypes = j.at("numOpTypes").get<int>();

			jobs.clear();
			for(const auto& item: j.at("jobs")) {
				Job job;
				job.id = item.at("id").get<int>();
				job.nextOpIndex = item.value("nextOpIndex", 0);		 // default to 0 if not present
				job.lastOpEndTime = item.value("lastOpEndTime", 0);	 // default to 0 if not present

				for(const auto& opItem: item.at("operations")) {
					Operation op;
					op.type = opItem.at("type").get<int>();
					op.eligibleMachines = opItem.at("eligibleMachines").get<std::vector<int>>();
					job.operations.push_back(op);
				}

				jobs.push_back(job);
			}

			processingTimes = j.at("processingTimes").get<std::vector<std::vector<int>>>();
		} catch(const json::exception& e) {
			throw std::runtime_error("JSON processing error: " + std::string(e.what()));
		}

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

	for(int o = 0; o < data.numOpTypes; ++o) {
		data.processingTimes[o].resize(data.numMachines, 0);

		// Create shuffled machine list for this operation type
		std::vector<int> machines(data.numMachines);
		std::iota(machines.begin(), machines.end(), 0);	 // fill vector with 0, 1, ..., numMachines-1
		std::random_shuffle(machines.begin(), machines.end());

		// Select 1-3 eligible machines with random times
		const int numEligibleMachines = data.opFlexibilityRange.first + rand() % flexibilityRangeRemainder;

		data.processingTimes[o].resize(data.numMachines, 0);  // Initialize all to 0 (ineligible)

		for(int m = 0; m < numEligibleMachines; ++m) {
			// random processing time for random eligible machine
			data.processingTimes[o][machines[m]] = data.opDurationRange.first + rand() % durationRangeRemainder;
			// TODO: MAKE SURE EACH MACHINE IS ELIGIBLE
		}
	}

	// 2. Generate jobs with unique operation types
	for(int j = 0; j < data.numJobs; ++j) {
		Job job;
		job.id = j;

		// Create shuffled operation types
		std::vector<int> opTypes(data.numOpTypes);
		std::iota(opTypes.begin(), opTypes.end(), 0);
		std::random_shuffle(opTypes.begin(), opTypes.end());

		// Select 5-10 unique operations
		const int numOperations = data.opCountPerJobRange.first + rand() % operationRangeRemainder;

		for(int o = 0; o < numOperations; ++o) {
			Operation operation;
			operation.type = opTypes[o];  // Random operation type

			// get eligible machines for this operation type (non zero in processingTimes)
			for(int m = 0; m < data.numMachines; ++m) {
				if(data.processingTimes[operation.type][m] > 0) {
					operation.eligibleMachines.push_back(m);
				}
			}

			job.operations.push_back(operation);
		}
		data.jobs.push_back(job);

		// TODO: fix eligible machines for each operation
	}

	return data;
}

/* GPU Implementation */

struct GPUOperation {
	int type;
	int* eligibleMachines;	// Pointer to array in device memory
	int eligibleCount;
	int* successorsIDs;
	int successorCount;
	int predecessorCount; // -1 = done, 0 = availible, 1+ = not availible yet
	int lastPredecessorEndTime;
};

struct GPUJob {
	int id;
	GPUOperation* operations;  // Device pointer
	int operationCount;
	//int nextOpIndex;
	//int lastOpEndTime;
};

struct GPUProblem {
	int numMachines;
	int numJobs;
	int numOpTypes;
	GPUJob* jobs;		   // Device pointer
	int* processingTimes;  // Flattened [opType][machine]
};

class JobShopDataGPU
{
public:
	static void FreeGPUData(GPUProblem& gpuProblem);
	// Upload CPU data to GPU
	static GPUProblem UploadToGPU(const JobShopData& problem);
	// Free GPU memory
	void DownloadFromGPU(GPUProblem& gpuProblem, JobShopData& cpuProblem);
};
#endif	// JOB_SHOP_DATA_H