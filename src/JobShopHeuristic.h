#ifndef JOB_SHOP_HEURISTIC_H
#define JOB_SHOP_HEURISTIC_H

#include "JobShopData.h"
#include "NeuralNetwork.h"
#include <vector>

class JobShopHeuristic {
public:
    JobShopHeuristic(const std::vector<int>& topology);

    struct Solution {
        double makespan;
        std::vector<std::vector<int>> schedule; // Harmonogram dla każdej maszyny
    };

    Solution Solve(const JobShopData& data);

private:
    NeuralNetwork neuralNetwork;

    std::vector<float> ExtractFeatures(const JobShopData& data, int jobId, int operationId, int machineId);
    void UpdateSchedule(JobShopData& data, int jobId, int operationId, int machineId, Solution& solution);
};

#endif // JOB_SHOP_HEURISTIC_H