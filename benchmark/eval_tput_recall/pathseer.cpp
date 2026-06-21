#include "../bencher/pathseer.hpp"
#include "../utils/logger.hpp"
#include "../utils/timing.hpp"

int main() {

    int top_k = 10;
    // int init_efSearch = top_k;
    int init_efSearch = 2;
    float min_tput = 10; // 10 qps
    std::string base_dir = "../../dataset/";
    std::string index_base_dir = "../../indexes/pathseer/";

    Timing timer;

    std::vector<std::string> dataset_array = {
        "SIFT1M", 
        "HM",
        "LAION",
        "Arxiv",
        "Paper",
        "Amazon",
    };


    std::vector<std::vector<int>> M_params_array = {
        {64},
        {16},
        {32},
        {64},
        {64},
        {64},
    };

    std::vector<int> M_exp_construct_params = {
        128,
        128,
        128,
        128,
        128,
        128
    };

    // different PathSeer realization

    std::vector<std::string> pathseer_realization = {
        "full-tech",
        // "only-expanding",
    };

    for (int ds_idx = 0; ds_idx < dataset_array.size(); ds_idx++) {

        if(ds_idx!=0)
            continue;

        std::string dataset = dataset_array[ds_idx];
        std::string dataset_base_dir = base_dir + dataset;

        std::vector<int> M_params = M_params_array[ds_idx];
        int M_exp_construct = M_exp_construct_params[ds_idx];

        pathseer_searchParameter search_param;

        std::string load_metadata_name = dataset_base_dir + "/load/load.meta";
        std::string query_metadata_name = dataset_base_dir + "/query/query.meta";
        std::string golden_file_name = dataset_base_dir + "/golden/golden_cosine_10.bin";

        for (auto using_method : pathseer_realization){
            
            // if(ds_idx==2 && using_method=="full-tech")
            //     continue;

            // int end_times = using_method=="full-tech" ? 3 : 1;
            for(int test=0; test<1; test++){
                for(int M : M_params){

                    // we may need to load the virtual overhead
                    int virtual_overhead = 0, dist_comp_ns = 0, filter_time_ns = 0;
                    float coef_a = 0, coef_b = 0;
                    // whether using the coef_mode
                    // bool using_coef_mode = true;
                    bool using_coef_mode = false;
                    std::string virtual_overhead_file_path = "../../results/pathseer_virtual_overhead_profilier/logs/" + dataset + "_pathseer_M_" + std::to_string(M) + "_nsleep_0.txt";

                    // std::vector<int>M_exp_search_params = {0, M/2};
                    // for(int _M = M; _M<=M_exp_construct; _M+=16){
                    //     M_exp_search_params.push_back(_M);
                    // }
                    std::vector<int>M_exp_search_params = {M/2};

                    for(int M_exp : M_exp_search_params){

                        // skip the auto parameter decision for all the other methods
                        if((using_method != "full-tech") && M_exp==0)
                            continue;

                        PathSeer_bencher bench(dataset_base_dir);

                        char buff[128];
                        if(using_method=="only-expanding")
                            sprintf(buff, "%d:pathseer-exp-only_%s_Mexp_%d_M_%d_MexpConstruct_%d", test, dataset.c_str(), M_exp, M, M_exp_construct);
                        else{
                            sprintf(buff, "%d:pathseer_%s_Mexp_%d_M_%d_MexpConstruct_%d", test, dataset.c_str(), M_exp, M, M_exp_construct);
                        }
                        Logger *logger = new Logger(std::string(buff));

                        if(using_method=="only-expanding")
                            sprintf(buff, "%s%s-exp-only_Mexp_%d_M_%d.idx", index_base_dir.c_str(), dataset.c_str(), M_exp_construct, M);
                        else
                            sprintf(buff, "%s%s_Mexp_%d_M_%d.idx", index_base_dir.c_str(), dataset.c_str(), M_exp_construct, M);
                        
                        printf("Index name: %s\n", buff);

                        bench.load_persist(buff);
                        bench.set_logger(logger);

                        std::string golden_path = golden_file_name;

                        bench.set_metadata(query_metadata_name);

                        // only load when Mexp=0
                        if(!M_exp){
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
                        }

                        int efSearch = init_efSearch;
                        search_param.efSearch = efSearch;
                        search_param.golden_file = golden_path;
                        search_param.top_k = top_k;
                        search_param.M_exp_search = M_exp;
                        search_param.load_metadata_path = load_metadata_name;

                        int stop_recall = (M_exp != 0 || using_method=="only-expanding") ? 90 : 99;


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

                        float multiplier = 1.5;
                        float recall_prev = bench.batch_query(&search_param);
                        int nr_tested = 0;
                        while(1){
                            efSearch = (int)(efSearch * multiplier);
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

                            if(recall >= stop_recall || recall - recall_prev < 0.1)
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

