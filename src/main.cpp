#include <cstdlib>
#include <ctime>
#include <iostream>

#include "JobShopData.cuh"
#include "JobShopHeuristic.cuh"

int main() {
	srand(time(0));

	const bool generateRandomJobs = false;
	const bool generateRandomNNSetup = false;

	const std::vector<int> topology = {4, 32, 16, 1};  // TODO: implement dynamic NN input size

	try {
		JobShopData data;

		if(generateRandomJobs) {
			data = GenerateData();
			data.SaveToJson("jobshop_data");
		} else {
			data.LoadFromJson("jobshop_data");
		}

		NeuralNetwork nn;

		if(generateRandomNNSetup) {
			nn = NeuralNetwork(topology); // Generate new
            nn.SaveToJson("weights_and_biases");
		} else {
            nn.LoadFromJson("weights_and_biases"); // Load existing
		}

		JobShopHeuristic heuristic(std::move(nn)); // Transfer ownership
		
		// JobShopHeuristic heuristic("weights_and_biases");

		JobShopHeuristic::CPUSolution solution = heuristic.Solve(data);

		// std::cout << "Makespan: " << solution.makespan << std::endl;

		heuristic.PrintSchedule(solution, data);

	} catch(const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}