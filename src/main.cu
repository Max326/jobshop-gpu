#include "main_cmaes.cuh"
#include <string>
#include <vector>

std::vector<std::string> datasets = {
    "rnd_JT(5)_J(15)_M(5)_JO(5-10)_O(20)_OM(1-3)",
    "rnd_JT(5)_J(15)_M(5)_JO(5-10)_O(20)_OM(2-5)",
    "rnd_JT(5)_J(15)_M(5)_JO(10-15)_O(20)_OM(2-5)",
    "rnd_JT(5)_J(15)_M(5)_JO(10-15)_O(20)_OM(1-3)",
    "rnd_JT(5)_J(15)_M(15)_JO(5-10)_O(20)_OM(1-3)",
    "rnd_JT(5)_J(15)_M(15)_JO(5-10)_O(20)_OM(2-5)",
    "rnd_JT(5)_J(15)_M(15)_JO(10-15)_O(20)_OM(1-3)",
    "rnd_JT(5)_J(15)_M(15)_JO(10-15)_O(20)_OM(2-5)",
    "rnd_JT(5)_J(30)_M(5)_JO(5-10)_O(20)_OM(1-3)",
    "rnd_JT(5)_J(30)_M(5)_JO(5-10)_O(20)_OM(2-5)",
    "rnd_JT(5)_J(30)_M(5)_JO(10-15)_O(20)_OM(1-3)",
    "rnd_JT(5)_J(30)_M(5)_JO(10-15)_O(20)_OM(2-5)",
    "rnd_JT(5)_J(30)_M(15)_JO(5-10)_O(20)_OM(1-3)",
    "rnd_JT(5)_J(30)_M(15)_JO(5-10)_O(20)_OM(2-5)",
    "rnd_JT(5)_J(30)_M(15)_JO(10-15)_O(20)_OM(1-3)",
    "rnd_JT(5)_J(30)_M(15)_JO(10-15)_O(20)_OM(2-5)",
    "rnd_JT(10)_J(15)_M(15)_JO(5-10)_O(20)_OM(1-3)",
    "rnd_JT(10)_J(15)_M(15)_JO(5-10)_O(20)_OM(2-5)",
    "rnd_JT(10)_J(15)_M(15)_JO(10-15)_O(20)_OM(1-3)",
    "rnd_JT(10)_J(15)_M(15)_JO(10-15)_O(20)_OM(2-5)",
    "rnd_JT(10)_J(30)_M(15)_JO(5-10)_O(20)_OM(1-3)",
    "rnd_JT(10)_J(30)_M(15)_JO(5-10)_O(20)_OM(2-5)",
    "rnd_JT(10)_J(30)_M(15)_JO(10-15)_O(20)_OM(1-3)",
    "rnd_JT(10)_J(30)_M(15)_JO(10-15)_O(20)_OM(2-5)",
    "rnd_JT(10)_J(15)_M(5)_JO(5-10)_O(20)_OM(1-3)",
    "rnd_JT(10)_J(15)_M(5)_JO(5-10)_O(20)_OM(2-5)",
    "rnd_JT(10)_J(15)_M(5)_JO(10-15)_O(20)_OM(1-3)",
    "rnd_JT(10)_J(15)_M(5)_JO(10-15)_O(20)_OM(2-5)",
    "rnd_JT(10)_J(30)_M(5)_JO(5-10)_O(20)_OM(1-3)",
    "rnd_JT(10)_J(30)_M(5)_JO(5-10)_O(20)_OM(2-5)",
    "rnd_JT(10)_J(30)_M(5)_JO(10-15)_O(20)_OM(1-3)",
    "rnd_JT(10)_J(30)_M(5)_JO(10-15)_O(20)_OM(2-5)"
};


int main(int argc, char *argv[]){
    int n = datasets.size();

    int start_index = 0;
    int ds_amount = 2; // Number of datasets to process in this run

    for (int i = start_index; i < start_index + ds_amount && i < n; ++i) {
        std::string train_problem_file = "TRAIN/" + datasets[i] + "_total.json";
        std::string validate_problem_file = "VALID/" + datasets[i] + "_validation.json";
        std::string test_problem_file = "TEST/" + datasets[i] + "_test.json";

        main_cmaes(train_problem_file, validate_problem_file, test_problem_file);
    }

    return 0;
}