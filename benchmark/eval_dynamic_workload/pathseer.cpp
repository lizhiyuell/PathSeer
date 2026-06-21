#include "../bencher/pathseer.hpp"
#include "../utils/logger.hpp"

int main() {

    int top_k = 10;
    int init_efSearch = top_k;
    std::string base_dir = "../../dataset/";
    std::string index_base_dir = "../../indexes/pathseer/";

    std::vector<std::string> dataset_array = {
        "Paper",
    };


    std::vector<std::vector<int>> M_params_array = {
        {64},
    };

    std::vector<int> M_exp_construct_params = {
        128
    };


    for (int ds_idx = 0; ds_idx < dataset_array.size(); ds_idx++) {

        std::string dataset = dataset_array[ds_idx];
        std::string dataset_base_dir = base_dir + dataset;

        std::vector<int> M_params = M_params_array[ds_idx];
        int M_exp_construct = M_exp_construct_params[ds_idx];

        pathseer_searchParameter search_param;

        for(int test=0; test<3; test++){
            for(int M : M_params){

                // we may need to load the virtual overhead
                int virtual_overhead = 0, dist_comp_ns = 0, filter_time_ns = 0;
                float coef_a = 0, coef_b = 0;
                // whether using the coef_mode
                bool using_coef_mode = false;
                int M_exp = 0;
                std::string virtual_overhead_file_path = "../../results/pathseer_virtual_overhead_profilier/logs/" + dataset + "_pathseer_M_" + std::to_string(M) + "_nsleep_0.txt";

                char buff[128];
                sprintf(buff, "%d:pathseer_%s_Mexp_%d_M_%d_MexpConstruct_%d", test, dataset.c_str(), M_exp, M, M_exp_construct);
                Logger *logger = new Logger(std::string(buff));

                for(int test_period=0; test_period<=50; test_period++){

                    std::string load_metadata_name = dataset_base_dir + "/load/load_" + std::to_string(test_period) + ".meta";
                    std::string query_metadata_name = dataset_base_dir + "/query/query_" + std::to_string(test_period) + ".meta";
                    std::string golden_file_name = dataset_base_dir + "/golden/golden_" + std::to_string(test_period) + "_cosine_10.bin";

                    logger->loggingf("[Period] %d\n", test_period);

                    PathSeer_bencher bench(dataset_base_dir);
                    sprintf(buff, "%s%s_Mexp_%d_M_%d.idx", index_base_dir.c_str(), dataset.c_str(), M_exp_construct, M);

                    bench.load_persist(buff);
                    bench.set_logger(logger);

                    std::string golden_path = golden_file_name;

                    bench.set_metadata(query_metadata_name);

                    bench.load_virtual_overhead(virtual_overhead_file_path, dist_comp_ns, filter_time_ns, virtual_overhead, coef_a, coef_b);
                    if(virtual_overhead){
                        search_param.virtual_overhead = virtual_overhead;
                        if(using_coef_mode){
                            search_param.coef_a = coef_a;
                            search_param.coef_b = coef_b;
                        }
                        bench.set_profiled_dist_comp_ns(dist_comp_ns);
                        bench.set_profiled_filter_ns(filter_time_ns);
                        logger->loggingf("[AutoProfile] using dist-comp-ns=%d filter-time-ns=%d virtual-overhead=%d coef_a=%f coef_b=%f, using_coef_mode=%d\n", dist_comp_ns, filter_time_ns, virtual_overhead, coef_a, coef_b, using_coef_mode);
                    }

                    int efSearch = init_efSearch;
                    search_param.efSearch = efSearch;
                    search_param.golden_file = golden_path;
                    search_param.top_k = top_k;
                    search_param.M_exp_search = M_exp;
                    search_param.load_metadata_path = load_metadata_name;

                    int stop_recall = 99;

                    if(dataset=="Paper"){
                        search_param.method = QUERY_METHOD_INT_LARGER;
                    }
                    else
                        assert(false);

                    std::vector<float> target_recalls = {85,90};

                    for(float target_recall : target_recalls) {
                        bench.auto_batch_query(&search_param, target_recall, 0, 19);
                    }

                    bench.clear();
                }
                delete logger;
            }
        }
    }
    return 0;
}

