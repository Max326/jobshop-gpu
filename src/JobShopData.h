#ifndef JOB_SHOP_DATA_H
#define JOB_SHOP_DATA_H

#pragma once

#include <vector>
#include <string>

struct Job {
    int id;
    std::vector<int> operations; // Lista typów operacji
};

struct Machine {
    int id;
};

struct JobShopData {
    int numMachines;
    int numJobs;
    int numOperations;
    std::vector<Job> jobs;
    std::vector<std::vector<int>> processingTimes; // Macierz czasu przetwarzania [operacja][maszyna]
};

inline JobShopData GenerateExampleData() {
    JobShopData data;
    data.numMachines = 5;
    data.numJobs = 10;
    data.numOperations = 10;

    // Inicjalizacja jobów
    for (int j = 0; j < data.numJobs; ++j) {
        Job job;
        job.id = j;
        for (int o = 0; o < 5 + rand() % 6; ++o) { // Od 5 do 10 operacji na job
            job.operations.push_back(rand() % data.numOperations); // Losowy typ operacji
        }
        data.jobs.push_back(job);
    }

    // Inicjalizacja macierzy czasu przetwarzania
    data.processingTimes.resize(data.numOperations, std::vector<int>(data.numMachines, 0));
    for (int o = 0; o < data.numOperations; ++o) {
        for (int m = 0; m < data.numMachines; ++m) {
            data.processingTimes[o][m] = 1 + rand() % 10; // Losowy czas przetwarzania od 1 do 10
        }
    }

    return data;
}

#endif // JOB_SHOP_DATA_H