#include <iostream>
#include <cstdlib>
#include <ctime>
#include "JobShopData.h"
#include "JobShopHeuristic.h"

JobShopData GenerateExampleData() {
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

int main() {
    srand(time(0));

    // Generowanie przykładowych danych
    JobShopData data = GenerateExampleData();

    // Konfiguracja heurystyki
    std::vector<int> topology = {3, 16, 1}; // Przykładowa topologia sieci
    JobShopHeuristic heuristic(topology);

    // Rozwiązanie problemu
    JobShopHeuristic::Solution solution = heuristic.Solve(data);

    // Wyświetlenie wyniku
    std::cout << "Makespan: " << solution.makespan << std::endl;

    return 0;
}