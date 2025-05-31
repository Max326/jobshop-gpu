#include "../libcmaes/cmaes.h"
#include <iostream>
#include "JobShopGPUEvaluator.cuh" 

using namespace libcmaes;

extern JobShopGPUEvaluator* g_gpu_evaluator;

class customCMAStrategy : public CMAStrategy<CovarianceUpdate>
{

public:
    customCMAStrategy(FitFunc &func,
                        CMAParameters<> &parameters)
        :CMAStrategy<CovarianceUpdate>(func,parameters) {}

    ~customCMAStrategy() {}

    dMat ask() {
        return CMAStrategy<CovarianceUpdate>::ask();
    }

    void eval(const dMat &candidates, const dMat &phenocandidates=dMat(0,0)) {  
                    
        int nn_count = candidates.cols();   // amount of neural networks in training
        int weights_count = candidates.rows();  // amount of weights per neural network
        Eigen::VectorXd fvalues(nn_count);  // vector of makespans for each nn (xd)
        
    
        std::cout << "candidates size: " << candidates.rows() << "x" << candidates.cols() << std::endl;
        // Modify begin

        // candidates.col(r) -> all weights of neural network of index r, r=(0,191)
        // ...
        // ...
        // Use neural networks to fill fvalues vector ! 
        // fvalues[r] = ... makespan form scheduling with neural network of index r
        // EXAMPLE BELOW \/
        // Konwersja dMat -> Eigen::MatrixXd
        Eigen::MatrixXd eCandidates(weights_count, nn_count);
        for (int i = 0; i < weights_count; i++) {
            for (int j = 0; j < nn_count; j++) {
                eCandidates(i, j) = candidates.coeff(i, j);
            }
        }
        //std::cout << "eCandidates: " << eCandidates << std::endl;

        fvalues = g_gpu_evaluator->EvaluateCandidates(eCandidates);
        //std::cout << "fvalues: " << fvalues.transpose() << std::endl;
        // ...
        // ...
        // fvalues -> avg makespan of neural network of index r, r=(0,191)

        // Modify End

        for (int r = 0; r < nn_count; ++r) {
            _solutions.get_candidate(r).set_x(candidates.col(r));
            _solutions.get_candidate(r).set_fvalue(fvalues[r]);         
        }
        update_fevals(candidates.cols());
    }

    void tell() {
        return CMAStrategy<CovarianceUpdate>::tell();
    }

    bool stop() {
        return CMAStrategy<CovarianceUpdate>::stop();
    }

    float get_best_fvalue() const {
        return _solutions.get_candidate(0).get_fvalue();
    }
};