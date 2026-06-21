#include "../bencher/pathseer.hpp"
#include "../utils/logger.hpp"
#include <chrono>



int main() {

    int top_k = 10;
    int init_efSearch = top_k;
    std::string base_dir = "../../dataset/";
    std::string index_base_dir = "../../indexes/pathseer/";

    std::vector<std::string> dataset_array = {
        "SIFT1M", 
        "HM",
        "LAION",
        "Arxiv",
        "Paper",
        "Amazon",
    };

    std::vector<std::vector<int>> M_params_array = {
        {16, 32},
        {32, 64},
        {16, 64},
        {16, 32},
        {16, 32},
        {16, 32},
    };

    std::vector<int> M_exp_construct_params = {
        128,
        128,
        128,
        128,
        128,
        128
    };

    std::vector<std::vector<int>> M_exp_search_params = {
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
    };

    for (int ds_idx = 0; ds_idx < dataset_array.size(); ds_idx++) {

        std::string dataset = dataset_array[ds_idx];
        std::string dataset_base_dir = base_dir + dataset;

        std::vector<int> M_params = M_params_array[ds_idx];
        int M_exp_construct = M_exp_construct_params[ds_idx];

        pathseer_searchParameter search_param;

        std::string load_metadata_name = dataset_base_dir + "/load/load.meta";
        std::string query_metadata_name = dataset_base_dir + "/query/query.meta";
        std::string golden_file_name = dataset_base_dir + "/golden/golden_cosine_10.bin";

        for(int test=0; test<3; test++){
            for(int M : M_params){

                // we may need to load the virtual overhead
                int virtual_overhead = 0, dist_comp_ns = 0, filter_time_ns = 0;
                float coef_a = 0, coef_b = 0;
                std::string virtual_overhead_file_path = "../../results/pathseer_virtual_overhead_profilier/logs/" + dataset + "_pathseer_M_" + std::to_string(M) + "_nsleep_0.txt";

                for(int M_exp : M_exp_search_params[ds_idx]){

                    PathSeer_bencher bench(dataset_base_dir);

                    char buff[128];
                    sprintf(buff, "%d:M-sensitivity_%s_Mexp_%d_M_%d_MexpConstruct_%d", test, dataset.c_str(), M_exp, M, M_exp_construct);
                    Logger *logger = new Logger(std::string(buff));

                    sprintf(buff, "%s%s_Mexp_%d_M_%d.idx", index_base_dir.c_str(), dataset.c_str(), M_exp_construct, M);
                    
                    bench.load_persist(buff);
                    bench.set_logger(logger);

                    std::string golden_path = golden_file_name;

                    bench.set_metadata(query_metadata_name);

                    // only load when Mexp=0
                    if(!M_exp){
                        bench.load_virtual_overhead(virtual_overhead_file_path, dist_comp_ns, filter_time_ns, virtual_overhead, coef_a, coef_b);
                        if(virtual_overhead){
                            search_param.virtual_overhead = virtual_overhead;
                            bench.set_profiled_dist_comp_ns(dist_comp_ns);
                            bench.set_profiled_filter_ns(filter_time_ns);
                            logger->loggingf("[AutoProfile] using dist-comp-ns=%d filter-time-ns=%d virtual-overhead=%d", dist_comp_ns, filter_time_ns, virtual_overhead, true);
                        }
                    }

                    int efSearch = init_efSearch;
                    search_param.efSearch = efSearch;
                    search_param.golden_file = golden_path;
                    search_param.top_k = top_k;
                    search_param.M_exp_search = M_exp;
                    search_param.load_metadata_path = load_metadata_name;

                    if(dataset=="Paper"){
                        search_param.method = QUERY_METHOD_INT_LARGER;
                    }
                    else if(dataset=="SIFT1M"){
                        search_param.method = QUERY_METHOD_INT_ARRAY;
                    }
                    else if(dataset=="HM"){
                        search_param.method = QUERY_METHOD_MULTI_FILTER;
                    }
                    else if (dataset=="LAION"){
                        search_param.method = QUERY_METHOD_REGEX;
                    }
                    else if (dataset=="Amazon"){
                        search_param.method = QUERY_METHOD_FLOAT_LOWER;
                    }
                    else if (dataset=="Arxiv"){
                        search_param.method = QUERY_METHOD_COMPOSITE_FILTER;
                    }
                    else
                        assert(false);

                    std::vector<float> target_recalls = {85, 90};

                    // find the lower and upper boundary
                    float target_lower = target_recalls[0];
                    float target_higher = target_recalls[target_recalls.size()-1];

                    int lower_efSearch = -1;
                    int higher_efSearch = -1;
                    float lower_recall = 0;
                    float higher_recall = 100;

                    float multiplier = 2;
                    float recall_prev = bench.batch_query(&search_param);
                    int nr_tested = 0;
                    while(1){
                        efSearch = (int)(efSearch * multiplier);
                        nr_tested++;
                        search_param.efSearch = efSearch;
                        float recall = bench.batch_query(&search_param);

                        if(recall < target_lower && (target_lower-recall) < (target_lower-lower_recall)){
                            lower_recall = recall;
                            lower_efSearch = efSearch;
                        }
                        else if(recall >= target_higher && (recall - target_higher) < (higher_recall - target_higher)){
                            higher_recall = recall;
                            higher_efSearch = efSearch;
                        }

                        if(recall >= 90 || recall - recall_prev < 0.1)
                            break;

                        recall_prev = recall;
                    }

                    for(float target_recall : target_recalls) {
                        // bench the result of each target recall
                        bench.auto_batch_query(&search_param, target_recall, lower_efSearch, higher_efSearch);
                    }

                    bench.clear();
                    delete logger;
                }
            }
        }
    }
    return 0;
}

