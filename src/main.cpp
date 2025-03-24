#include <iostream>
#include <cstdlib>
#include <ctime>
#include "JobShopData.h"
#include "JobShopHeuristic.h"

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

    heuristic.neuralNetwork.SaveToJson("weights_and_biases.json");

    return 0;
}