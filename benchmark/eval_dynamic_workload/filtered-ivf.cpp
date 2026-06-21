#include "../bencher/filtered_ivf.hpp"
#include "../utils/logger.hpp"

int main() {
    int top_k = 10;
    int init_nprobe = 1;
    std::string base_dir = "../../dataset/";
    std::string index_base_dir = "../../indexes/filtered_ivf/";

    std::vector<std::string> dataset_array = {
        "Paper",
    };

    std::vector<int> N_params_array = {512};

    for(int ds_idx = 0; ds_idx < dataset_array.size(); ds_idx++){

        std::string dataset = dataset_array[ds_idx];
        std::string dataset_base_dir = base_dir + dataset;

        filtered_ivf_searchParameter search_param;

        int N = N_params_array[ds_idx];

        for(int test=0; test<3; test++){

            char buff[128];
            sprintf(buff, "%d:filtered-ivf_%s_N_%d", test, dataset.c_str(), N);
            Logger *logger = new Logger(std::string(buff));


            for(int test_period=0; test_period<=50; test_period++){

                std::string load_metadata_name = dataset_base_dir + "/load/load_" + std::to_string(test_period) + ".meta";
                std::string query_metadata_name = dataset_base_dir + "/query/query_" + std::to_string(test_period) + ".meta";
                std::string golden_file_name = dataset_base_dir + "/golden/golden_" + std::to_string(test_period) + "_cosine_10.bin";

                logger->loggingf("[Period] %d\n", test_period);

                FILTERED_IVF_bencher bench(dataset_base_dir);

                sprintf(buff, "%s%s_N_%d.idx", index_base_dir.c_str(), dataset.c_str(), N);
                bench.load_persist(buff);//load index
                bench.set_logger(logger);

                bench.set_metadata(query_metadata_name);

                int nprobe = init_nprobe;
                search_param.nprobe = nprobe;
                search_param.cluster = N;//nlist
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
                    // bench the result
                    bench.auto_batch_query(&search_param, target_recall, 1, 10);
                }

                bench.clear();
            }

            delete logger;
        }
    }
    return 0;
}