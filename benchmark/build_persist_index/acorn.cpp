#include "../bencher/acorn.hpp"
#include "../utils/logger.hpp"
#include <cstdlib>

int main() {

    std::string base_dir = "../../dataset/";
    std::string index_base_dir = "../../indexes/acorn/";
    system(("mkdir -p " + index_base_dir).c_str());

    Logger logger("", true);

    std::vector<std::string> dataset_array = {
        // "SIFT1M",
        // "SIFT1M_max_2",
        // "SIFT1M_max_4",
        // "SIFT1M_max_8",
        // "SIFT1M_max_16",
        // "SIFT1M_max_32",
        // "SIFT1M_max_64",
        // "Paper",
        // "HM",
        // "LAION",
        // "Amazon",
        // "Arxiv",
        "SIFT1M_max_1",
        "SIFT1M_max_10",
        "SIFT1M_max_100",
        "SIFT1M_max_1000",
    };

    std::vector<std::vector<int>> gamma_params_array = {
        // {12},
        // {2},
        // {4},
        // {8},
        // {16},
        // {32},
        // {64},
        // {40},
        // {156},
        // {38},
        // {3},
        // {189},
        {2}, // we avoid using gamma=1
        {10},
        {100},
        {100}, // we avoid using a very large gamma for speed consideration
    };

    for (int i=0; i<dataset_array.size(); i++) {

        std::string dataset = dataset_array[i];

        logger.logging("---------- [dataset] " + dataset + " ----------", true);

        std::string dataset_base_dir = base_dir + dataset;

        ACORN_bencher bench(dataset_base_dir);
        bench.set_logger(&logger);

        std::vector<int> M_params = {16, 32, 64};
        std::vector<int> gamma_params = gamma_params_array[i];
        // This is the multipliers of M
        std::vector<int> M_beta_multiplier_params = {1, 2};

        acorn_buildParameter build_param;

        for(int M_beta_multiplier : M_beta_multiplier_params){
            for(int gamma : gamma_params){
                for(int M : M_params){

                    build_param.M = M;
                    build_param.gamma = gamma;
                    build_param.M_beta = M_beta_multiplier * M;

                    logger.logging("##### [BuildParams] M=" + std::to_string(build_param.M) + " gamma=" + std::to_string(build_param.gamma) + " M=" + std::to_string(build_param.M) + " M_beta=" + std::to_string(build_param.M_beta) + " #####", true);

                    bench.build(&build_param);

                    // save the index
                    char buff[128];
                    sprintf(buff, "%s%s_Mbeta_%d_M_%d_gamma_%d.idx", index_base_dir.c_str(), dataset.c_str(), M_beta_multiplier * M, M, gamma);
                    bench.save_persist(buff);
                    printf("Persist finish at: %s\n", buff);
                    bench.clear();
                }
            }
        }
    }
    return 0;
}
