#include "customCMAES.hpp"
#include "JobShopGPUEvaluator.cuh"

/*
:0 will be used to mark OG Stanislaus's code regarding cmaes, 
so that i (or Copilot) dont fuck it up 
*/
using namespace libcmaes;

JobShopGPUEvaluator* g_gpu_evaluator = nullptr;

int main(int argc, char *argv[])
{
    const std::vector<int> topology = {101, 32, 16, 1};
    const int batch_size = 50;
    const std::string problem_file = "test_10k.json";
    int population_size = 192;//:0

    int nn_weights_count = NeuralNetwork::CalculateTotalParameters(topology);//:0
    
    std::cout<< nn_weights_count <<std::endl;
    
    JobShopGPUEvaluator gpu_evaluator(problem_file, topology);
    g_gpu_evaluator = &gpu_evaluator;

    int total_problems = gpu_evaluator.GetTotalProblems();

    std::vector<double> x0(nn_weights_count, 0.0);
/*     for(int i = 0; i < nn_weights_count; i++) {
        x0[i] = (double)rand() / RAND_MAX * 0.01 - 0.005;
        
    } */
    
    double sigma = 0.1;//:0
    CMAParameters<> cmaparams(x0, sigma, population_size);//:0  
    
    //const uint64_t fixed_seed = 12345; // Choose any constant value
    //CMAParameters<> cmaparams(x0, sigma, population_size, fixed_seed);

    FitFunc eval = [](const double *x, const int N) -> double { return 0.0; }; //:0
    ESOptimizer<customCMAStrategy,CMAParameters<>> optim(eval, cmaparams);//:0

    int batch_start = 0;

    while(!optim.stop() && gpu_evaluator.SetCurrentBatch(batch_start, batch_size)) {
        dMat candidates = optim.ask();//:0
        optim.eval(candidates);//:0
        optim.tell();//:0
        optim.inc_iter();//:0
        batch_start += batch_size;
    }
    //std::cout << optim.get_solutions() << std::endl;//:0
    return 0;
}