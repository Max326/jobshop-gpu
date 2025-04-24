#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "JobShopData.cuh"
#include "JobShopHeuristic.cuh"

// TODO: implement more features, according to the Flexible Job Shop document
// 		- wasted time
// 		- total number of operations left
// 		- one hot encodings
// TODO: test scheduling correctness
// TODO: parallel operations (graphs)

int main() {
	srand(time(0));

	const bool generateRandomJobs = false;
	const bool generateRandomNNSetup = false;
	const int numProblems = 9600;

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

		auto solutions_batch = SolutionManager::CreateGPUSolutions(numProblems, data.numMachines, 100);

		GPUProblem* d_problems;
		std::vector<GPUProblem> h_problems(numProblems);

		GPUProblem template_problem = JobShopDataGPU::UploadToGPU(data);  // TODO cleanup
		for(int i=0; i<numProblems; ++i) {
			h_problems[i] = JobShopDataGPU::UploadToGPU(data);  // Create unique copy for each problem
		}

		// Copy template to all problems
		cudaMalloc(&d_problems, sizeof(GPUProblem) * numProblems);

		std::vector<GPUProblem> temp(numProblems, template_problem);
		cudaMemcpy(d_problems, h_problems.data(),
				   sizeof(GPUProblem) * numProblems,
				   cudaMemcpyHostToDevice);

		// 4. Create heuristic solver
		JobShopHeuristic heuristic(std::move(nn));
		
		// 5. Solve on GPU (even though we're just doing one problem)
		heuristic.SolveBatch(d_problems, &solutions_batch, numProblems);

		cudaError_t kernelErr = cudaGetLastError();
		if(kernelErr != cudaSuccess) {
			std::cerr << "Kernel error: " << cudaGetErrorString(kernelErr) << "\n";
		}

		// 6. Download results
		JobShopHeuristic::CPUSolution* solutions = new JobShopHeuristic::CPUSolution[numProblems];

		for(int i = 0; i < numProblems; ++i) {
			solutions[i].FromGPU(solutions_batch, i);
		}

		heuristic.PrintSchedule(solutions[0], data);

		// 7. Clean up GPU memory
		SolutionManager::FreeGPUSolutions(solutions_batch);
		cudaFree(d_problems);

		JobShopDataGPU::FreeGPUData(template_problem);
		for (int i = 0; i<numProblems; ++i) {
			JobShopDataGPU::FreeGPUData(h_problems[i]);
		}

		delete[] solutions;

	} catch(const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}