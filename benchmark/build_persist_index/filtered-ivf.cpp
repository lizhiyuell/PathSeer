#include "../bencher/filtered_ivf.hpp"
#include "../utils/logger.hpp"
#include <cstdlib>

int main() {

    std::string base_dir = "../../dataset/";
    std::string index_base_dir = "../../indexes/filtered_ivf/";
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

        if(dataset!="LAION")
            continue;

        logger.logging("---------- [dataset] " + dataset + " ----------", true);

        std::string dataset_base_dir = base_dir + dataset;
    
        FILTERED_IVF_bencher bench(dataset_base_dir);
        bench.set_logger(&logger);

        std::vector<int> N_params = {256, 512, 1024, 2048};

        filtered_ivf_buildParameter build_param;

        for(int N : N_params){

            build_param.nlist = N;
            logger.logging("##### [BuildParams] N=" + std::to_string(build_param.nlist) + " #####", true);

            bench.build(&build_param);

            // save the index
            char buff[128];
            sprintf(buff, "%s%s_N_%d.idx", index_base_dir.c_str(), dataset.c_str(), N);
            bench.save_persist(buff);
            printf("Persist finish at: %s\n", buff);
    
            bench.clear();
        }
    }
    return 0;
}
