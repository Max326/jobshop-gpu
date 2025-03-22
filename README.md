# jobshop-gpu

This is a repository for computing Neural Network outputs on a GPU

## Compilation and running

To compile:
```
cd src
nvcc -o NeuralNetwork main.cu NeuralNetwork.cu
```

To run:
```
./NeuralNetwork
```