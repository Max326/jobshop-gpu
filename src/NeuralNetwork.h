#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>

class NeuralNetwork
{
public:
	NeuralNetwork(const std::vector<int> &topology);
	void Forward(const std::vector<float> &input, std::vector<float> &output);

private:
	std::vector<int> topology;
	std::vector<std::vector<float>> weights;
	std::vector<std::vector<float>> biases;
};

#endif	// NEURAL_NETWORK_H