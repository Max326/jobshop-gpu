#include "../libcmaes/cmaes.h"
#include <iostream>

using namespace libcmaes;

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
        Eigen::VectorXd fvalues(nn_count);  // vector of makespans for each nn
        
        // Modify begin

        // candidates.col(r) -> all weights of neural network of index r, r=(0,191)
        // ...
        // ...
        // Use neural networks to fill fvalues vector ! 
        // fvalues[r] = ... makespan form scheduling with neural network of index r
        // EXAMPLE BELOW \/
        for (int r = 0; r < nn_count; ++r) {
            fvalues[r] = candidates.coeff(0, r) * candidates.coeff(0, r) - 2.0 * candidates.coeff(0, r);
        }
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
};