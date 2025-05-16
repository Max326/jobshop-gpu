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
#include <map>
#include <algorithm>
#include <chrono>

#include "FileManager.h"

// Host-side operation structure
struct Operation {
    int type;
    std::vector<int> eligibleMachines;   // List of machines that can perform this operation
    std::vector<int> successorsIDs;      // Indices of successor operations
    int predecessorCount = 0;            // Number of predecessors
    int lastPredecessorEndTime = 0;
};

// Host-side job structure
struct Job {
    int id;
    int type;
    std::vector<Operation> operations;   // Ordered list of operations
    int nextOpIndex = 0;                 // Next operation to schedule
    int lastOpEndTime = 0;               // End time of last operation
};

// Main host-side data structure
class JobShopData
{
public:
    using json = nlohmann::json;

    int numMachines = 0;
    int numJobs = 0;
    int numOpTypes = 0;
    std::vector<Job> jobs;
    std::vector<std::vector<int>> processingTimes; // [opType][machine]
    std::vector<std::unordered_set<int>> machineEligibleOperations; // [machine] -> set of opTypes

    // Optional config fields (used for data generation or validation)
    bool operInJobMultiplication = false;
    std::pair<int, int> jobCountRange;
    std::pair<int, int> opDurationRange;
    std::pair<int, int> opCountPerJobRange;
    std::pair<int, int> opFlexibilityRange;


    // Load multiple instances from a JSON array file
    static std::vector<JobShopData> LoadFromParallelJson(const std::string& filename, int numInstances) {
		using clock = std::chrono::high_resolution_clock;
		std::vector<JobShopData> result;
		std::string full_path = FileManager::GetFullPath(filename);
	
		// Efficient file read
		auto t_file_start = clock::now();
		std::ifstream in(full_path, std::ios::binary | std::ios::ate);
		if(!in) throw std::runtime_error("Failed to open file: " + full_path);
		size_t size = in.tellg();
		in.seekg(0);
		std::string buffer(size, '\0');
		in.read(buffer.data(), size);
		auto t_file_end = clock::now();
		std::cout << "[DIAG] File read: "
				  << std::chrono::duration_cast<std::chrono::milliseconds>(t_file_end - t_file_start).count()
				  << " ms\n";
	
		// Parse from memory
		auto t_parse_start = clock::now();
		nlohmann::json j_array = nlohmann::json::parse(buffer);
		auto t_parse_end = clock::now();
		std::cout << "[DIAG] JSON parse: "
				  << std::chrono::duration_cast<std::chrono::milliseconds>(t_parse_end - t_parse_start).count()
				  << " ms\n";
	
		if(!j_array.is_array())
			throw std::runtime_error("JSON root is not an array as expected.");
		if(j_array.size() < numInstances)
			throw std::runtime_error("Not enough instances in file.");
	
		// Convert JSON to JobShopData
		auto t_conv_start = clock::now();
		for(int i = 0; i < numInstances; ++i) {
			JobShopData data;
			const auto& j = j_array[i];
			data.numMachines = j["numM"].get<int>();
			data.numJobs = j["numJ"].get<int>();
			data.numOpTypes = j["numO"].get<int>();
	
			data.jobs.clear();
			const auto& jsonJobs = j["Jobs"];
			const auto& jsonPrec = j["Prec"];
	
			std::map<std::vector<std::pair<int, int>>, int> opTypeMap;
            std::map<std::vector<std::vector<std::pair<int, int>>>, int> jobTypeMap;

			int currentOpType = 0;
            int currentJobType = 0;
	
			for (size_t jobIdx = 0; jobIdx < jsonJobs.size(); ++jobIdx) {
                Job job;
                job.id = jobIdx;
                const auto& jsonOperations = jsonJobs[jobIdx];
                std::vector<std::vector<std::pair<int, int>>> jobSignature; // Vector to store the machine-time pairs for the job
    
                for (const auto& jsonOp : jsonOperations) {
                    Operation op;
                    std::vector<std::pair<int, int>> machineTimes;
                    for (const auto& mt : jsonOp) {
                        int time = mt[0].get<int>();
                        int machine = mt[1].get<int>();
                        op.eligibleMachines.push_back(machine);
                        machineTimes.emplace_back(machine, time);
                    }
                    std::sort(machineTimes.begin(), machineTimes.end());
                    auto it = opTypeMap.find(machineTimes);
                    if (it == opTypeMap.end()) {
                        opTypeMap[machineTimes] = currentOpType;
                        op.type = currentOpType++;
                    }
                    else {
                        op.type = it->second;
                    }
                    job.operations.push_back(op);
                    jobSignature.push_back(machineTimes); // Add the machine-time pairs to the job signature
                }
    
                auto jobIt = jobTypeMap.find(jobSignature);
                if (jobIt == jobTypeMap.end()) {
                    jobTypeMap[jobSignature] = currentJobType;
                    job.type = currentJobType++;
                }
                else {
                    job.type = jobIt->second;
                }
    
                data.jobs.push_back(job);
            }
	
			data.processingTimes.assign(currentOpType, std::vector<int>(data.numMachines, -1));
			for(const auto& pair_type: opTypeMap) {
				const auto& mtPairs = pair_type.first;
				int type_id = pair_type.second;
				for(const auto& pair_machine_time: mtPairs) {
					int machine = pair_machine_time.first;
					int time = pair_machine_time.second;
					data.processingTimes[type_id][machine] = time;
				}
			}
	
			for(size_t jobIdx = 0; jobIdx < jsonPrec.size(); ++jobIdx) {
				auto& job = data.jobs[jobIdx];
				const auto& jobPrec = jsonPrec[jobIdx];
				for(size_t opIdx = 0; opIdx < jobPrec.size(); ++opIdx) {
					auto& op = job.operations[opIdx];
					const auto& predecessors = jobPrec[opIdx];
					op.predecessorCount = predecessors.size();
					for(int predIdx: predecessors) {
						job.operations[predIdx].successorsIDs.push_back(opIdx);
					}
				}
			}
			data.numOpTypes = currentOpType;
			data.BuildMachineEligibleOperations();
			result.push_back(std::move(data));
		}
		auto t_conv_end = clock::now();
		std::cout << "[DIAG] JobShopData conversion: "
				  << std::chrono::duration_cast<std::chrono::milliseconds>(t_conv_end - t_conv_start).count()
				  << " ms\n";
	
		return result;
	}

    // Build machineEligibleOperations from jobs/operations
    void BuildMachineEligibleOperations() {
        machineEligibleOperations.clear();
        machineEligibleOperations.resize(numMachines);
        for(const auto& job : jobs) {
            for(const auto& op : job.operations) {
                for(int machineId : op.eligibleMachines) {
                    machineEligibleOperations[machineId].insert(op.type);
                }
            }
        }
    }

    // Validate loaded data
    bool Validate() const {
        if(numMachines <= 0 || numJobs <= 0 || numOpTypes <= 0) return false;
        if(processingTimes.size() != numOpTypes) return false;
        for(const auto& row: processingTimes) {
            if(row.size() != numMachines) return false;
        }
        return true;
    }
};

// GPU-side operation structure
struct GPUOperation {
    int type;
    int eligibleMachinesOffset; // Offset in eligibleMachines array
    int eligibleCount;
    int successorsOffset;       // Offset in successorsIDs array
    int successorCount;
    int predecessorCount;
    int lastPredecessorEndTime;
};

// GPU-side job structure
struct GPUJob {
    int id;
    int type;
    int operationsOffset; // Offset in operations array
    int operationCount;
};

// GPU-side problem structure
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

// Batch structure for GPU upload
struct BatchJobShopGPUData {
    std::vector<GPUJob> jobs;
    std::vector<GPUOperation> operations;
    std::vector<int> eligibleMachines;
    std::vector<int> successorsIDs;
    std::vector<int> processingTimes;
    std::vector<int> jobsOffsets;         // size: numProblems+1
    std::vector<int> operationsOffsets;   // size: numProblems+1
    std::vector<int> eligibleOffsets;     // size: numProblems+1
    std::vector<int> successorsOffsets;   // size: numProblems+1
    std::vector<int> processingTimesOffsets; // size: numProblems+1
    std::vector<GPUProblem> gpuProblems;
    int numProblems = 0;
};

// GPU data manager
class JobShopDataGPU
{
public:
    static BatchJobShopGPUData PrepareBatchCPU(const std::vector<JobShopData>& problems);
    static void UploadBatchToGPU(
        BatchJobShopGPUData& batch,
        GPUProblem*& d_gpuProblems,
        GPUJob*& d_jobs,
        GPUOperation*& d_ops,
        int*& d_eligible,
        int*& d_succ,
        int*& d_procTimes,
        int& numProblems);
    static void FreeBatchGPUData(GPUProblem* d_gpuProblems,
        GPUJob* d_jobs, GPUOperation* d_ops,
        int* d_eligible, int* d_succ, int* d_procTimes);
    void DownloadFromGPU(GPUProblem& gpuProblem, JobShopData& cpuProblem);
};

#endif	// JOB_SHOP_DATA_CUH