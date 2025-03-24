#include <cstdlib>
#include <ctime>
#include <iostream>

#include "JobShopData.h"
#include "JobShopHeuristic.h"

int main() {
	srand(time(0));

	bool generateRandomJobs = false;
	bool generateRandomNNSetup = false;

	try {
        JobShopData data;
		// Generowanie przykładowych danych
		if(generateRandomJobs) {
			data = GenerateData();
			data.SaveToJson("jobshop_data.json");
		} else {
            data.LoadFromJson("../data/jobshop_data.json");
        }

		// Konfiguracja heurystyki
		std::vector<int> topology = {3, 16, 1};	 // Przykładowa topologia sieci

		if(generateRandomNNSetup) {
			NeuralNetwork exampleNeuralNetwork(topology);
			exampleNeuralNetwork.SaveToJson("weights_and_biases.json");
		}

		// JobShopHeuristic heuristic(topology);
		JobShopHeuristic heuristic("../data/weights_and_biases.json");

		// Rozwiązanie problemu
		JobShopHeuristic::Solution solution = heuristic.Solve(data);

		// Wyświetlenie wyniku
		std::cout << "Makespan: " << solution.makespan << std::endl;

		// heuristic.neuralNetwork.SaveToJson("weights_and_biases.json");
	} catch(const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}