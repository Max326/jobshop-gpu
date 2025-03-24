#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>

class NeuralNetwork
{
public:
	NeuralNetwork(const std::vector<int>& topology,
				  const std::vector<std::vector<float>>* weights = nullptr,
				  const std::vector<std::vector<float>>* biases = nullptr);

	void Forward(const std::vector<float>& input, std::vector<float>& output);

	void SaveToJson(const std::string& filename) const;

	void LoadFromJson(const std::string& filename);

private:
	std::vector<int> topology;
	std::vector<std::vector<float>> weights;
	std::vector<std::vector<float>> biases;
};

#endif	// NEURAL_NETWORK_H