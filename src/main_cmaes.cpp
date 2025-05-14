#include "customCMAES.hpp"

using namespace libcmaes;

// int main(int argc, char *argv[])
// {
//     int population_size = 192;
//     int nn_weights_count = 2; // ilosc wag w sieci np. 300                                                                          
//     std::vector<double> x0(nn_weights_count, 0.0);
//     double sigma = 0.1;

//     CMAParameters<> cmaparams(x0, sigma, population_size);  
//     FitFunc eval = [](const double *x, const int N) -> double { return 0.0; };
//     ESOptimizer<customCMAStrategy,CMAParameters<>> optim(eval,cmaparams);

//     while(!optim.stop()) {
//         dMat candidates = optim.ask();
//         optim.eval(candidates);
//         optim.tell();
//         optim.inc_iter(); // important step: signals next iteration.                                              
//     }
//     std::cout << optim.get_solutions() << std::endl;
// }