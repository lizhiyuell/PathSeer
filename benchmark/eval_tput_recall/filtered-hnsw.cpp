#include "../bencher/filtered_hnsw.hpp"
#include "../utils/logger.hpp"
#include "../utils/timing.hpp"

int main() {

    int top_k = 10;
    int init_efSearch = top_k;
    float min_tput = 10; // 10 qps
    std::string base_dir = "../../dataset/";
    std::string index_base_dir = "../../indexes/filtered_hnsw/";

    Timing timer;

    std::vector<std::string> dataset_array = {
        "SIFT1M", 
        "HM",
        "LAION",
        "Paper",
        "Arxiv",
        "Amazon",
    };

    std::vector<int> M_params_array = {32, 64, 64, 32, 64, 64};

    for(int ds_idx = 0; ds_idx < dataset_array.size(); ds_idx++){

        if(ds_idx!=4)
            continue;

        std::string dataset = dataset_array[ds_idx];
        std::string dataset_base_dir = base_dir + dataset;

        filtered_hnsw_searchParameter search_param;

        int M = M_params_array[ds_idx];

        std::string load_metadata_name = dataset_base_dir + "/load/load.meta";
        std::string query_metadata_name = dataset_base_dir + "/query/query.meta";
        std::string golden_file_name = dataset_base_dir + "/golden/golden_cosine_10.bin";

        for(int test=0; test<3; test++){

            Filtered_HNSW_bencher bench(dataset_base_dir);

            char buff[128];
            sprintf(buff, "%d:filtered-hnsw_%s_M_%d", test, dataset.c_str(), M);
            Logger *logger = new Logger(std::string(buff));

            sprintf(buff, "%s%s_M_%d.idx", index_base_dir.c_str(), dataset.c_str(), M);
            bench.load_persist(buff);
            bench.set_logger(logger);

            bench.set_metadata(query_metadata_name);

            int efSearch = init_efSearch;
            search_param.efSearch = efSearch;
            search_param.golden_file = golden_file_name;
            search_param.top_k = top_k;
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
            // find the lower and upper boundary
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

                int64_t start_time = timer.get_s_time();
                float recall = bench.batch_query(&search_param);
                int64_t end_time = timer.get_s_time();

                if(recall < target_lower && (target_lower-recall) < (target_lower-lower_recall)){
                    lower_recall = recall;
                    lower_efSearch = efSearch;
                }
                else if(recall >= target_higher && (recall - target_higher) < (higher_recall - target_higher)){
                    higher_recall = recall;
                    higher_efSearch = efSearch;
                }

                // early stop due to slow test and enough recall rate
                if(recall > target_higher && (end_time - start_time)!=0 && (1.0 * bench.get_nr_query() / (end_time - start_time)) < min_tput){
                    break;
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

    return 0;
}
