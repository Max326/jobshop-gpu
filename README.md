# jobshop-gpu

This is a repository for computing Neural Network outputs on a GPU

## Compilation and running

To compile:

```bash
cd src
nvcc -o fjssp main.cpp NeuralNetwork.cu JobShopHeuristic.cpp
```

To run:

```bash
./fjssp

```
