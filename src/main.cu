#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "JobShopData.cuh"
#include "JobShopHeuristic.cuh"

int main() {
	srand(time(0));

	const bool generateRandomJobs = false;
	const bool generateRandomNNSetup = false;
	const int numProblems = 5000;	// Start with 1 problem for testing

	const std::vector<int> topology = {4, 32, 16, 1};

	try {
		// 1. Load or generate problem data
		JobShopData data;
		if(generateRandomJobs) {
			data = GenerateData();
			data.SaveToJson("jobshop_data");
		} else {
			data.LoadFromJson("jobshop_data");
		}

		// 2. Load or generate neural network
		NeuralNetwork nn;
		if(generateRandomNNSetup) {
			nn = NeuralNetwork(topology);
			nn.SaveToJson("weights_and_biases");
		} else {
			nn.LoadFromJson("weights_and_biases");
		}

		// 3. Prepare GPU data and upload to GPU
		GPUProblem gpuProblems[numProblems];
		SolutionManager::GPUSolution gpuSolutions[numProblems];

		for(int i = 0; i < numProblems; ++i) {
			gpuProblems[i] = JobShopDataGPU::UploadToGPU(data);
			// b) Create GPU solution container
			gpuSolutions[i] = SolutionManager::CreateGPUSolution(data.numMachines, 100);  // ! 100 ops per machine max -- this needs to be the same in JopShopHeuristic.cuh
		}

		// 4. Create heuristic solver
		JobShopHeuristic heuristic(std::move(nn));

		// 5. Solve on GPU (even though we're just doing one problem)
		heuristic.SolveBatch(gpuProblems, gpuSolutions, numProblems);

		cudaError_t kernelErr = cudaGetLastError();
		if(kernelErr != cudaSuccess) {
			std::cerr << "Kernel error: " << cudaGetErrorString(kernelErr) << "\n";
		}

		// 6. Download and display results
		JobShopHeuristic::CPUSolution solutions[numProblems];

		for(int i = 0; i < numProblems; ++i) {
			solutions[i].FromGPU(gpuSolutions[i]);
		}

		heuristic.PrintSchedule(solutions[0], data);

		// 7. Clean up GPU memory
		for(int i = 0; i < numProblems; ++i) {
			SolutionManager::FreeGPUSolution(gpuSolutions[i]);
			JobShopDataGPU::FreeGPUData(gpuProblems[i]);
		}

	} catch(const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}