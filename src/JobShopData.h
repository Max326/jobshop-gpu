#ifndef JOB_SHOP_DATA_H
#define JOB_SHOP_DATA_H

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

#endif // JOB_SHOP_DATA_H