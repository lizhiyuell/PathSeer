#include "../bencher/navix.hpp"
#include "../utils/logger.hpp"
#include "../utils/timing.hpp"

int main() {
    int top_k = 10;
    // int init_efSearch = top_k;
    int init_efSearch = 2;
    float min_tput = 10; // 10 qps
    std::string base_dir = "../../dataset/";
    std::string index_base_dir = "../../indexes/navix/";

    Timing timer;

    std::vector<std::string> dataset_array = {
        "Paper",
    };

    std::vector<int> M_params_array = {32};

    for(int ds_idx = 0; ds_idx < dataset_array.size(); ds_idx++){

        std::string dataset = dataset_array[ds_idx];
        std::string dataset_base_dir = base_dir + dataset;

        navix_searchParameter search_param;

        int M = M_params_array[ds_idx];

        for(int test=0; test<3; test++){

            char buff[128];
            sprintf(buff, "%d:navix_%s_M_%d", test, dataset.c_str(), M);
            Logger *logger = new Logger(std::string(buff));

            for(int test_period=0; test_period<=50; test_period++){

                std::string load_metadata_name = dataset_base_dir + "/load/load_" + std::to_string(test_period) + ".meta";
                std::string query_metadata_name = dataset_base_dir + "/query/query_" + std::to_string(test_period) + ".meta";
                std::string golden_file_name = dataset_base_dir + "/golden/golden_" + std::to_string(test_period) + "_cosine_10.bin";

                logger->loggingf("[Period] %d\n", test_period);

                NaviX_bencher NaviX_bencher(dataset_base_dir);


                sprintf(buff, "%s%s_M_%d.idx", index_base_dir.c_str(), dataset.c_str(), M);
                NaviX_bencher.load_persist(buff);
                NaviX_bencher.set_logger(logger);
                NaviX_bencher.set_metadata(query_metadata_name);

                int efSearch = init_efSearch;
                search_param.efSearch = efSearch;
                search_param.golden_file = golden_file_name;
                search_param.top_k = top_k;
                search_param.load_metadata_path = load_metadata_name;

                if(dataset=="Paper"){
                    search_param.method = QUERY_METHOD_INT_LARGER;
                }
                else
                    assert(false);
                    
                std::vector<float> target_recalls = {85,90};

                for(float target_recall : target_recalls) {
                    NaviX_bencher.auto_batch_query(&search_param, target_recall, 1, 31);
                }

                NaviX_bencher.clear();
            }
            delete logger;
        }
    }
    return 0;
}