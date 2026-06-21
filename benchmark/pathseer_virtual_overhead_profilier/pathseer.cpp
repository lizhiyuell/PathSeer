// for the evaluated workloads, search a suitable virtual overhead bound for each of them

#include "../bencher/pathseer.hpp"
#include "../utils/logger.hpp"
#include <chrono>


int main() {

    int top_k = 10;
    int init_efSearch = top_k;
    std::string base_dir = "../../dataset/";
    std::string index_base_dir = "../../indexes/pathseer/";

    std::vector<std::string> dataset_array = {
        "SIFT1M", // sleep = 0
        "SIFT1M", // sleep = 200
        "HM",
        "LAION",
        "Paper",
        "Arxiv",
        "Amazon",
    };

    std::vector<int> sleep_array = {
        0,
        28,
        0,
        0,
        0,
        0,
        0
    };

    // record all pairs of M1 and M2 for each dataset
    std::vector<std::vector<std::vector<int>>> param_lists = {
    
        // SIFT1M
        {
            {64, 128},
        },
        // SIFT1M-sleep-only
        {
            {64, 128},
        },
        // HM
        {
            {16, 128},
        },
        // LAION
        {
            {32, 128}, // M1 sensitivity
        },
        // Paper
        {
            {64, 128},
        },
        // Arxiv
        {
            {64, 128},
        },
        // Amazon
        {
            {64, 128},
        },
    };

    for (int ds_idx = 0; ds_idx < dataset_array.size(); ds_idx++) {

        if(ds_idx!=0)
            continue;

        for(auto it : param_lists[ds_idx]){

            int M = it[0];
            int M_exp_construct = it[1];
            int sleep_val = sleep_array[ds_idx];

            // the range for searched parameters
            std::vector<int> M_exp_search_params;
            // push half of the M additionally
            M_exp_search_params.push_back(M/2);
            for(int Mexp=M; Mexp<=M_exp_construct; Mexp+=16){
                M_exp_search_params.push_back(Mexp);
            }

            // if ordinary setting has too little Mexp, we reconf it
            if(M_exp_search_params.size() <= 2){
                M_exp_search_params.clear();
                int delta = (M_exp_construct - M/2) / 4;
                assert(delta > 1);
                for(int Mexp = M/2; Mexp<= M_exp_construct; Mexp += delta)
                    M_exp_search_params.push_back(Mexp);
                
                if(M_exp_search_params[M_exp_search_params.size()-1]!=M_exp_construct)
                    M_exp_search_params.push_back(M_exp_construct);
            }

            std::string dataset = dataset_array[ds_idx];
            std::string dataset_base_dir = base_dir + dataset;
            char buff[128];
            sprintf(buff, "pathseer_%s_M_%d_MexpConstruct_%d_sleep_%d", dataset.c_str(), M, M_exp_construct, sleep_val);
            Logger *logger = new Logger(std::string(buff));

            PathSeer_bencher bench(dataset_base_dir);

            sprintf(buff, "%s%s_Mexp_%d_M_%d.idx", index_base_dir.c_str(), dataset.c_str(), M_exp_construct, M);
            bench.load_persist(buff);
            bench.set_logger(logger);

            // this is only used for profiling the template overhead. we don't assume the query metadata is known before the query starts
            bench.set_metadata(dataset_base_dir + "/query/query.meta");

            enum query_method qmethod;
            if(dataset=="Paper"){
                qmethod = QUERY_METHOD_INT_LARGER;
            }
            else if(dataset=="SIFT1M"){
                if(sleep_val)
                    qmethod = QUERY_METHOD_INT_ARRAY_SLEEP;
                else
                    qmethod = QUERY_METHOD_INT_ARRAY;
            }
            else if(dataset=="HM"){
                qmethod = QUERY_METHOD_MULTI_FILTER;
            }
            else if (dataset=="LAION"){
                qmethod = QUERY_METHOD_REGEX;
            }
            else if (dataset=="Amazon"){
                qmethod = QUERY_METHOD_FLOAT_LOWER;
            }
            else if (dataset=="Arxiv"){
                qmethod = QUERY_METHOD_COMPOSITE_FILTER;
            }
            else
                assert(false);

            logger->loggingf("[Profile_virtual_overhead_bound] dataset=%s sleep_val=%d\n", dataset.c_str(), sleep_val);

            bench.profile_virtual_overhead_bound(qmethod, sleep_val, M_exp_search_params);

            bench.clear();
            delete logger;
        }

    }
    return 0;
}
