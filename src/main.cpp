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

		std::vector<int> topology = {2, 32, 16, 1};	 // Przykładowa topologia sieci

		// NeuralNetwork nn(topology, &weights, &biases);

		NeuralNetwork nnFiller(topology);

		if(generateRandomNNSetup) {
			nnFiller.SaveToJson("weights_and_biases");
			// JobShopHeuristic heuristic(std::move(nnFiller));
		} else {
			nnFiller.LoadFromJson("weights_and_biases");
		}
		JobShopHeuristic heuristic("weights_and_biases");

		JobShopHeuristic::Solution solution = heuristic.Solve(data);

		// std::cout << "Makespan: " << solution.makespan << std::endl;

		heuristic.PrintSchedule(solution, data);

	} catch(const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}