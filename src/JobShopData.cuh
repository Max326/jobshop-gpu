#ifndef JOB_SHOP_DATA_CUH
#define JOB_SHOP_DATA_CUH

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
	std::vector<int> successorsIDs;
	int predecessorCount = 0;  // -1 = done, 0 = available, 1+ = not available yet
	int lastPredecessorEndTime = 0;
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
	std::vector<std::vector<int>> processingTimes;					 // processing times for each operation type on each machine
	std::vector<std::unordered_set<int>> machineEligibleOperations;	 // eligible operations for each machine

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

				// Preserve but ignore legacy fields
				job.nextOpIndex = item.value("nextOpIndex", 0);
				job.lastOpEndTime = item.value("lastOpEndTime", 0);

				// Load operations with linear dependencies
				auto operationsJson = item.at("operations");
				for(size_t opIdx = 0; opIdx < operationsJson.size(); opIdx++) {
					Operation op;
					op.type = operationsJson[opIdx].at("type").get<int>();
					op.eligibleMachines = operationsJson[opIdx].at("eligibleMachines").get<std::vector<int>>();

					// Set dependencies based on position
					op.predecessorCount = (opIdx == 0) ? 0 : 1;

					// Linear successors: each op points to next in sequence
					if(opIdx < operationsJson.size() - 1) {
						op.successorsIDs.push_back(opIdx + 1);
					}

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

	void LoadFromParallelJson(const std::string& filename, int instanceIndex = 0) {	 // Added instanceIndex parameter
		std::string full_path = FileManager::GetFullPath(filename);
		std::ifstream in(full_path);
		if(!in) throw std::runtime_error("Failed to open file: " + full_path);

		nlohmann::json j_array;	 // Parse the whole file content
		try {
			in >> j_array;
		} catch(nlohmann::json::parse_error& e) {
			throw std::runtime_error("Failed to parse JSON: " + std::string(e.what()));
		}

		// --- FIX START ---
		// Check if the parsed JSON is an array and if the requested index is valid
		if(!j_array.is_array()) {
			throw std::runtime_error("JSON root is not an array as expected.");
		}
		if(j_array.empty()) {
			throw std::runtime_error("JSON array is empty, no instances to load.");
		}
		if(instanceIndex < 0 || instanceIndex >= j_array.size()) {
			throw std::runtime_error("Invalid instance index requested: " + std::to_string(instanceIndex));
		}

		// Select the specific instance object from the array
		const auto& j = j_array[instanceIndex];
		// --- FIX END ---

		// Ensure the selected element is an object
		if(!j.is_object()) {
			throw std::runtime_error("Selected element at index " + std::to_string(instanceIndex) + " is not a JSON object.");
		}

		// 1. Parse basic metadata from the selected instance object
		numMachines = j["numM"].get<int>();
		numJobs = j["numJ"].get<int>();
		// numOpTypes from the file isn't strictly needed as the code calculates
		// the actual number of unique types later based on machine-time pairs.
		// We read it, but 'currentType' will determine the processingTimes size.
		numOpTypes = j["numO"].get<int>();

		// 2. Parse jobs and their operations
		jobs.clear();
		const auto& jsonJobs = j["Jobs"];
		const auto& jsonPrec = j["Prec"];  // Precedence constraints

		// Validate that the number of jobs in the data matches numJobs metadata
		if(jsonJobs.size() != numJobs) {
			throw std::runtime_error("Mismatch: numJ (" + std::to_string(numJobs) + ") != actual number of jobs found (" + std::to_string(jsonJobs.size()) + ")");
		}
		if(jsonPrec.size() != numJobs) {
			throw std::runtime_error("Mismatch: numJ (" + std::to_string(numJobs) + ") != actual number of precedence entries found (" + std::to_string(jsonPrec.size()) + ")");
		}

		// Map to track unique operation types (machine-time combinations)
		std::map<std::vector<std::pair<int, int>>, int> opTypeMap;
		int currentType = 0;  // This will count the actual unique operation types found

		for(size_t jobIdx = 0; jobIdx < jsonJobs.size(); ++jobIdx) {
			Job job;
			job.id = jobIdx;  // Assign job ID based on its position

			const auto& jsonOperations = jsonJobs[jobIdx];
			for(const auto& jsonOp: jsonOperations) {
				Operation op;
				std::vector<std::pair<int, int>> machineTimes;

				// Parse eligible machines and processing times
				for(const auto& mt: jsonOp) {
					// JSON format is [time, machine]
					int time = mt[0].get<int>();
					int machine = mt[1].get<int>();
					// Validate machine index
					if(machine < 0 || machine >= numMachines) {
						throw std::runtime_error("Invalid machine index " + std::to_string(machine) + " for job " + std::to_string(jobIdx));
					}
					op.eligibleMachines.push_back(machine);
					machineTimes.emplace_back(machine, time);  // Store as (machine, time) for sorting/mapping
				}

				if(machineTimes.empty()) {
					throw std::runtime_error("Operation with no machine options found for job " + std::to_string(jobIdx));
				}

				// Create unique operation type based on machine-time pairs
				std::sort(machineTimes.begin(), machineTimes.end());  // Sort by machine index first
				auto it = opTypeMap.find(machineTimes);
				if(it == opTypeMap.end()) {
					opTypeMap[machineTimes] = currentType;
					op.type = currentType++;
				} else {
					op.type = it->second;
				}

				job.operations.push_back(op);
			}

			// Store the job first to get correct operation indices for precedence processing later
			jobs.push_back(job);
		}

		// 3. Initialize processing times matrix based on discovered types
		processingTimes.assign(currentType, std::vector<int>(numMachines, -1));	 // Use -1 or another indicator for infeasible
		for(const auto& pair_type: opTypeMap) {
			const auto& mtPairs = pair_type.first;	// The sorted vector of (machine, time) pairs
			int type_id = pair_type.second;			// The assigned unique type ID
			for(const auto& pair_machine_time: mtPairs) {
				int machine = pair_machine_time.first;
				int time = pair_machine_time.second;
				processingTimes[type_id][machine] = time;
			}
		}

		// 4. Parse precedence constraints
		for(size_t jobIdx = 0; jobIdx < jsonPrec.size(); ++jobIdx) {
			auto& job = jobs[jobIdx];  // Get the already created job
			const auto& jobPrec = jsonPrec[jobIdx];

			// Validate that precedence array size matches the number of operations for the job
			if(jobPrec.size() != job.operations.size()) {
				throw std::runtime_error("Mismatch in precedence array size for job " + std::to_string(jobIdx) +
										 ": expected " + std::to_string(job.operations.size()) +
										 ", got " + std::to_string(jobPrec.size()));
			}

			for(size_t opIdx = 0; opIdx < jobPrec.size(); ++opIdx) {
				auto& op = job.operations[opIdx];			// Reference to the current operation
				const auto& predecessors = jobPrec[opIdx];	// Array of predecessor indices

				// Set predecessors for current operation
				op.predecessorCount = predecessors.size();

				// Update successors of predecessors
				for(int predIdx: predecessors) {
					// Validate predecessor index
					if(predIdx < 0 || predIdx >= job.operations.size()) {
						throw std::runtime_error("Invalid predecessor index " + std::to_string(predIdx) + " for operation " +
												 std::to_string(opIdx) + " in job " + std::to_string(jobIdx));
					}
					job.operations[predIdx].successorsIDs.push_back(opIdx);
				}
			}
		}

		// Optional: Call your validation function if it exists
		// if(!Validate()) {
		// 	throw std::runtime_error("Loaded data failed validation");
		// }
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

			operation.predecessorCount = (o == 0) ? 0 : 1;
			if(o < numOperations - 1) {
				operation.successorsIDs.push_back(o + 1);
			}

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
    int eligibleMachinesOffset; // offset 
    int eligibleCount;
    int successorsOffset;       // offset 
    int successorCount;
    int predecessorCount;  // -1 = done, 0 = available, 1+ = not available yet
    int lastPredecessorEndTime;
};

struct GPUJob {
    int id;
    int operationsOffset; // offset 
    int operationCount;
};

struct GPUProblem {
    int numMachines;
    int numJobs;
    int numOpTypes;
    GPUJob* jobs;                // Device pointer
    GPUOperation* operations;    // Device pointer 
    int* eligibleMachines;       // Device pointer 
    int* successorsIDs;          // Device pointer 
    int* processingTimes;        // Flattened [opType][machine]
};

class JobShopDataGPU
{
public:
	static void FreeGPUData(GPUProblem& gpuProblem);
	// Upload CPU data to GPU
	static GPUProblem UploadToGPU(const JobShopData& problem);
	static GPUProblem UploadParallelToGPU(const JobShopData& problem);
	// Free GPU memory
	void DownloadFromGPU(GPUProblem& gpuProblem, JobShopData& cpuProblem);
};
#endif	// JOB_SHOP_DATA_CUH