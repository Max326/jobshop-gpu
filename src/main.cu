#include "NeuralNetwork.h"
#include <iostream>

int main()
{
    // Przykładowa topologia: warstwa wejściowa (2 neurony), ukryta (3 neurony), wyjściowa (1 neuron)
    std::vector<int> topology = {6, 3, 1};
    NeuralNetwork network(topology);

    // Przykładowe dane wejściowe
    std::vector<float> input = {1.0f, 2.5f, 3.0f, -0.5f, -0.6f, 1.3f};
    std::vector<float> output;

    // Propagacja w przód
    network.Forward(input, output);

    // Wynik
    std::cout << "Output: " << output[0] << std::endl;

    return 0;
}