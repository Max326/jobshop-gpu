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

        if(generateRandomJobs) {
			data = GenerateData();
			data.SaveToJson("jobshop_data");
		} else {
            data.LoadFromJson("jobshop_data");
        }

		// Konfiguracja heurystyki
		// std::vector<int> topology = {3, 16, 1};	 // Przykładowa topologia sieci
		std::vector<int> topology = {3, 2, 1};	 // Przykładowa topologia sieci


		if(generateRandomNNSetup) {
			NeuralNetwork exampleNeuralNetwork(topology);
			exampleNeuralNetwork.SaveToJson("weights_and_biases");
		}

		// JobShopHeuristic heuristic(topology);
		JobShopHeuristic heuristic("weights_and_biases_test");

		// Rozwiązanie problemu
		JobShopHeuristic::Solution solution = heuristic.Solve(data);
		std::cout << "Makespan: " << solution.makespan << std::endl;

		// heuristic.neuralNetwork.SaveToJson("weights_and_biases");
	} catch(const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}