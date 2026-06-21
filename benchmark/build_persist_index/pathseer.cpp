#include "../bencher/pathseer.hpp"
#include "../utils/logger.hpp"
#include <omp.h>
#include <cstdlib>

int main() {

    std::string base_dir = "../../dataset/";
    std::string index_base_dir = "../../indexes/pathseer/";
    system(("mkdir -p " + index_base_dir).c_str());

    // array of all benchmarked dataset
    std::vector<std::string> dataset_array = {
        "Paper",
        "SIFT1M",
        "HM",
        "LAION",
        "Amazon",
        "Arxiv"
    };

    // top array: different dataset
    // second array: different building parameter
    // third array: parameter list [M1, M2, only_exp, opt_build]
    std::vector<std::vector<std::vector<int>>> param_lists{
        // Paper
        {
            {64, 128, 0, 1},
        },
        // SIFT1M
        {
            {64, 128, 0, 1},
        },
        // HM
        {
            {16, 128, 0, 1},
        },
        // LAION
        {
            {32, 128, 0, 1},
        },
        // Amazon
        {
            {64, 128, 0, 1},
        },
        // Arxiv
        {
            {64, 128, 0, 1},
        },
    };

    char buff[128];
    Logger *logger = new Logger("", true);

    for (int idx=0; idx<dataset_array.size(); idx++) {

        if(idx!=1)
            continue;

        std::string dataset = dataset_array[idx];

        logger->logging("---------- [dataset] " + dataset + " ----------", true);

        std::string dataset_base_dir = base_dir + dataset;

        PathSeer_bencher bench(dataset_base_dir);
        bench.set_logger(logger);

        pathseer_buildParameter build_param;

        // benchmark each parameter
        for(auto param : param_lists[idx]){
            int M = param[0];
            int M_exp = param[1];
            bool only_expanding = param[2];
            bool opt_build = param[3];
            build_param.M = M;
            build_param.M_exp = M_exp;
            build_param.efConstruction = M * 2;
            build_param.efConstructionExp = M_exp * 2;
            build_param.use_only_expanding = only_expanding;
            build_param.use_optimized_building = opt_build;

            logger->loggingf("##### [BuildParams] M=%d efConstruction=%d efConstructionExp=%d only_exp=%d optimized_building=%d #####\n", build_param.M, build_param.efConstruction, build_param.efConstructionExp, only_expanding, opt_build);

            bench.build(&build_param);
            
            // save the index
            if(only_expanding)
                sprintf(buff, "%s%s-exp-only_Mexp_%d_M_%d.idx", index_base_dir.c_str(), dataset.c_str(), M_exp, M);
            else{
                if(opt_build)
                    sprintf(buff, "%s%s_Mexp_%d_M_%d.idx", index_base_dir.c_str(), dataset.c_str(), M_exp, M);
                else
                    sprintf(buff, "%s%s-slow-build_Mexp_%d_M_%d.idx", index_base_dir.c_str(), dataset.c_str(), M_exp, M);
            }

            bench.save_persist(buff);
            printf("Persist finish at: %s\n", buff);

            bench.clear();
        }

    }
    delete logger;
    return 0;
}
