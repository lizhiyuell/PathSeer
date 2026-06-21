#include "../bencher/navix.hpp"
#include "../utils/logger.hpp"
#include <cstdlib>

int main() {
    std::string base_dir = "../../dataset/";
    std::string index_base_dir = "../../indexes/navix/";
    system(("mkdir -p " + index_base_dir).c_str());

    Logger logger("", true);

    std::string dataset_array [] = {
        "Paper",
        "SIFT1M",
        "HM",
        "LAION",
        "Amazon",
        "Arxiv"
    };

    for (auto dataset : dataset_array) {

        logger.logging("---------- [dataset] " + dataset + " ----------", true);

        std::string dataset_base_dir = base_dir + dataset;
    
        NaviX_bencher bench(dataset_base_dir);
        bench.set_logger(&logger);

        std::vector<int> M_params = {16, 32, 64};

        navix_buildParameter build_param;

        for(int M : M_params){
            build_param.M = M;
            build_param.efConstruction = M * 2;

            logger.logging("##### [BuildParams] M=" + std::to_string(build_param.M) + " efConstruction=" + std::to_string(build_param.efConstruction) + " #####", true);

            bench.build(&build_param);

            // save the index
            char buff[128];
            sprintf(buff, "%s%s_M_%d.idx", index_base_dir.c_str(), dataset.c_str(), M);
            bench.save_persist(buff);
            printf("Persist finish at: %s\n", buff);
    
            bench.clear();
        }
    }
    return 0;
}
