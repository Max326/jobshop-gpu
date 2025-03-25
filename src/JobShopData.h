#ifndef JOB_SHOP_DATA_H
#define JOB_SHOP_DATA_H

#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "FileManager.h"

struct Job {
	int id;
	std::vector<int> operations;  // Lista typów operacji
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
	int numOperations;
	std::vector<Job> jobs;
	std::vector<std::vector<int>> processingTimes;	// Macierz czasu przetwarzania [operacja][maszyna]

	void SaveToJson(const std::string& filename) const {
		FileManager::EnsureDataDirExists();
		std::string full_path = FileManager::GetFullPath(filename);

		json j;
		j["numMachines"] = numMachines;
		j["numJobs"] = numJobs;
		j["numOperations"] = numOperations;

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
		numOperations = j["numOperations"];

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
		if(numMachines <= 0 || numJobs <= 0 || numOperations <= 0) return false;
		if(processingTimes.size() != numOperations) return false;
		for(const auto& row: processingTimes) {
			if(row.size() != numMachines) return false;
		}
		return true;
	}
};

inline JobShopData GenerateData() {
	JobShopData data;
	data.numMachines = 15;
	data.numJobs = 30;
	data.numOperations = 10;

	// Inicjalizacja jobów
	for(int j = 0; j < data.numJobs; ++j) {
		Job job;
		job.id = j;
		for(int o = 0; o < 5 + rand() % 6; ++o) {					// Od 5 do 10 operacji na job
			job.operations.push_back(rand() % data.numOperations);	// Losowy typ operacji
		}
		data.jobs.push_back(job);
	}

	// Inicjalizacja macierzy czasu przetwarzania
	data.processingTimes.resize(data.numOperations, std::vector<int>(data.numMachines, 0));
	for(int o = 0; o < data.numOperations; ++o) {
		for(int m = 0; m < data.numMachines; ++m) {
			data.processingTimes[o][m] = 1 + rand() % 10;  // Losowy czas przetwarzania od 1 do 10
		}
	}

	return data;
}

#endif	// JOB_SHOP_DATA_H