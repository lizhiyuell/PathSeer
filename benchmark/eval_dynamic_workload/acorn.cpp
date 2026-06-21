#include "../bencher/acorn.hpp"
#include "../utils/logger.hpp"
#include <faiss/impl/IDSelector.h>

int main() {

    int top_k = 10;
    int init_efSearch = top_k;
    std::string base_dir = "../../dataset/";
    std::string index_base_dir = "../../indexes/acorn/";

    std::vector<std::string> dataset_array = {
        "Paper",
    };

    // parameters for different datasets
    std::vector<int> gamma_params_array = {40};
    std::vector<int> M_beta_multiplier_params_array = {2};
    std::vector<int> M_params_array = {32};

    for(int ds_idx = 0; ds_idx < dataset_array.size(); ds_idx++){

        std::string dataset = dataset_array[ds_idx];
        std::string dataset_base_dir = base_dir + dataset;

        int M = M_params_array[ds_idx];
        int gamma = gamma_params_array[ds_idx];
        int M_beta_multiplier = M_beta_multiplier_params_array[ds_idx];

        acorn_searchParameter search_param;

        for(int test=0; test<3; test++){
            char buff[128];
            sprintf(buff, "%d:acorn_%s_Mbeta_%d_M_%d_gamma_%d", test, dataset.c_str(), M_beta_multiplier*M, M, gamma);

            Logger *logger = new Logger(std::string(buff));

            for(int test_period=0; test_period<=50; test_period++){

                std::string load_metadata_name = dataset_base_dir + "/load/load_" + std::to_string(test_period) + ".meta";
                std::string query_metadata_name = dataset_base_dir + "/query/query_" + std::to_string(test_period) + ".meta";
                std::string golden_file_name = dataset_base_dir + "/golden/golden_" + std::to_string(test_period) + "_cosine_10.bin";

                logger->loggingf("[Period] %d\n", test_period);

                ACORN_bencher bench(dataset_base_dir);
                sprintf(buff, "%s%s_Mbeta_%d_M_%d_gamma_%d.idx", index_base_dir.c_str(), dataset.c_str(), M_beta_multiplier * M, M, gamma);
                bench.load_persist(buff);
                bench.set_logger(logger);

                bench.set_metadata(query_metadata_name);

                int efSearch = init_efSearch;
                search_param.golden_file = golden_file_name;
                search_param.top_k = top_k;
                search_param.efSearch = efSearch;
                search_param.load_metadata_path = load_metadata_name;
                search_param.filter_opt = false;

                logger->loggingf("[Filtered_opt] %d\n", search_param.filter_opt);

                if(dataset=="Paper"){
                    search_param.method = QUERY_METHOD_INT_LARGER;
                }
                else
                    assert(false);

                std::vector<float> target_recalls = {85,90};

                for(float target_recall : target_recalls) {
                    bench.auto_batch_query(&search_param, target_recall, top_k, 100);
                }

                bench.clear();
            }
            delete logger;
        }

    }
    return 0;
}
