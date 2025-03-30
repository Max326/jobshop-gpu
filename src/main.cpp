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

		if(generateRandomJobs) {  // TODO: fix data generation (don't generate in build)
			data = GenerateData();
			data.SaveToJson("jobshop_data");
		} else {
			data.LoadFromJson("jobshop_data");
		}

		// Konfiguracja heurystyki
		// std::vector<int> topology = {3, 16, 1};	 // Przykładowa topologia sieci

		std::vector<int> topology = {2, 32, 16, 1};	 // Przykładowa topologia sieci

		// nnFiller.SaveToJson("weights_and_biases");

		// std::vector<std::vector<float>> weights = {
		// 	{0.1, 0.2, 0.3, 0.4},
		// 	{0.5, 0.6},
		// };

		// std::vector<std::vector<float>> biases = {
		// 	{0.4, 0.5},
		// 	{0.6},
		// };

		// NeuralNetwork nn(topology, &weights, &biases);

		// nnFiller.weights = weights;
		// nnFiller.biases = biases;

		// NeuralNetwork nn(nnFiller.topology, &nnFiller.weights, &nnFiller.biases);

		// std::vector<float> output = nn.Forward({0.1, 0.2});

		// std::cout << "NN Result: " << output[0] << std::endl;

		// if(generateRandomNNSetup) {
		// 	NeuralNetwork exampleNeuralNetwork(topology);
		// 	exampleNeuralNetwork.SaveToJson("weights_and_biases");
		// 	// exampleNeuralNetwork.LoadFromJson("weights_and_biases");
		// }

		// JobShopHeuristic heuristic(topology);

		NeuralNetwork nnFiller(topology);

		if(generateRandomNNSetup) {
			nnFiller.SaveToJson("weights_and_biases");
			// JobShopHeuristic heuristic(std::move(nnFiller));
		} else {
			nnFiller.LoadFromJson("weights_and_biases");
		}
		JobShopHeuristic heuristic("weights_and_biases");

		// JobShopHeuristic heuristic("weights_and_biases");

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