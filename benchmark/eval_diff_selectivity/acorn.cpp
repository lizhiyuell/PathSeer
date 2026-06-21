#include "../bencher/acorn.hpp"
#include "../utils/logger.hpp"
#include <faiss/impl/IDSelector.h>

int main() {

    int top_k = 10;
    int init_efSearch = top_k;
    std::string base_dir = "../../dataset/";
    std::string index_base_dir = "../../indexes/acorn/";

    std::vector<std::string> dataset_array = {
        "SIFT1M_max_2",
        "SIFT1M_max_4",
        "SIFT1M_max_8",
        "SIFT1M_max_16",
        "SIFT1M_max_32",
        "SIFT1M_max_64",
        "SIFT1M_max_1",
        "SIFT1M_max_10",
        "SIFT1M_max_100",
        "SIFT1M_max_1000",
    };


    // parameters for different datasets
    std::vector<std::vector<int>> gamma_params_array = {
        {2},
        {4},
        {8},
        {16},
        {32},
        {64},
        {2},
        {10},
        {100},
        {100}
    };

    std::vector<std::vector<int>> M_beta_multiplier_params_array = {
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
    };

    int sleep_ns_array[] = {
        0,
        28, // 200ns
    };

    for(int ds_idx = 0; ds_idx < dataset_array.size(); ds_idx++){;

        std::string dataset = dataset_array[ds_idx];
        std::string dataset_base_dir = base_dir + dataset;

        // Iterate over load parameters.
        std::vector<int> gamma_params = gamma_params_array[ds_idx];
        std::vector<int> M_beta_multiplier_params = M_beta_multiplier_params_array[ds_idx];

        acorn_searchParameter search_param;

        std::string load_metadata_name = dataset_base_dir + "/load/load.meta";
        std::string query_metadata_name = dataset_base_dir + "/query/query.meta";
        std::string golden_file_name = dataset_base_dir + "/golden/golden_cosine_10.bin";

        for(int test=0; test<3; test++){
            for(int sleep_ns : sleep_ns_array){
                for(int M_beta_multiplier : M_beta_multiplier_params){
                    for(int gamma : gamma_params){
                        int M = 64;
                        char buff[128];
                        sprintf(buff, "%d:acorn_%s_Mbeta_%d_M_%d_gamma_%d_sleep_%d", test, dataset.c_str(), M_beta_multiplier*M, M, gamma, sleep_ns);
                        Logger *logger = new Logger(std::string(buff));

                        ACORN_bencher bench(dataset_base_dir);
                        sprintf(buff, "%s%s_Mbeta_%d_M_%d_gamma_%d.idx", index_base_dir.c_str(), dataset.c_str(), M_beta_multiplier * M, M, gamma);
                        bench.load_persist(buff);
                        bench.set_logger(logger);

                        bench.set_metadata(query_metadata_name);

                        int efSearch = init_efSearch;
                        search_param.golden_file = golden_file_name;
                        search_param.top_k = top_k;
                        search_param.efSearch = efSearch;
                        search_param.filter_opt = false;
                        search_param.load_metadata_path = load_metadata_name;

                        if(sleep_ns){
                            search_param.method = QUERY_METHOD_INT_ARRAY_SLEEP;
                            search_param.sleep_ns = sleep_ns;
                            search_param.filter_opt = true;
                        }
                        else
                            search_param.method = QUERY_METHOD_INT_ARRAY;

                        std::vector<float> target_recalls = {85, 90};

                        // find the lower and upper boundary
                        float target_lower = target_recalls[0];
                        float target_higher = target_recalls[target_recalls.size()-1];
                        int lower_efSearch = -1;
                        int higher_efSearch = -1;
                        float lower_recall = 0;
                        float higher_recall = 100;

                        float recall_prev = bench.batch_query(&search_param);

                        int nr_tested = 0;
                        while(1){
                            efSearch = (int)(efSearch * 1.5);
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

                            if(recall - recall_prev < 0.1 || recall >= 99)
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

    }
    return 0;
}
