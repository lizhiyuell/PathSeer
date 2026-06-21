#ifndef __ACORN_HPP__
#define __ACORN_HPP__

#include "bencher.hpp"
#include <faiss/IndexFlat.h>
#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>
#include <iostream>
#include <string>
#include <cassert>
#include <stdio.h>
#include <faiss/impl/IDSelector.h>
#include "../utils/logger.hpp"

typedef struct{
    size_t M;
    size_t gamma;
    size_t M_beta;
} acorn_buildParameter;

typedef struct{
    std::string golden_file;
    int top_k;
    int efSearch;
    enum query_method method=QUERY_NO_FILTER;
    int sleep_ns = 0;
    bool limited_mode = false;
    int target_idx = -1; // maybe we only want to query a certain idx
    bool filter_opt = false; // whether use the filter opt
    std::string load_metadata_path; // path for dataset metadata
    bool output_search = false;
} acorn_searchParameter;

class ACORN_bencher : public Bencher {
private:
    // Search index.
    faiss::IndexACORNFlat *index {nullptr};

    // Variable definitions.
    float *vector_data {nullptr}, *query_data {nullptr};
    char *query_bit_vector {nullptr};
    uint32_t nr_vector, dimension, nr_query, nr_bytes;

    std::vector<int> metadata;

    std::string base_dir;

public:
    ACORN_bencher(std::string base_dir, std::string metadata_name) {
        if(logger)
            logger->logging("--- Testing multibench_ACORN of " + base_dir + ": " + metadata_name + " ---\n", true);
        else
            printf("--- Testing multibench_ACORN of %s:%s ---\n", base_dir.c_str(), metadata_name.c_str());
        // Hybrid search: Load vector data and query data, as well as metadata, in bv form
        load_ann_data_bv_tight(
            base_dir, metadata_name,
            &vector_data, &nr_vector, &dimension,
            &query_data, &nr_query, &nr_bytes, &query_bit_vector,
            true
        );
    }

    ACORN_bencher(std::string base_dir) {
        if(logger)
            logger->logging("--- Testing multibench_ACORN of " + base_dir + " ---\n", true);
        else
            printf("--- Testing multibench_ACORN of %s ---\n", base_dir.c_str());
        // Hybrid search: Load vector data and query data, as well as metadata, in bv form
        this->base_dir = base_dir;
        load_ann_data(
            base_dir,
            &vector_data, &nr_vector, &dimension,
            &query_data, &nr_query,
            true
        );
    }

    void set_metadata_bv(std::string metadata_class){
        if(query_bit_vector)
            delete query_bit_vector;
        uint32_t this_nr_query;
        load_ann_metadata_bv_tight(this->base_dir, metadata_class, &this_nr_query, &nr_bytes, &query_bit_vector);
        assert(nr_query==this_nr_query);
    }

    // Destructor.
    ~ACORN_bencher() {
        delete vector_data;
        delete query_data;
        delete query_bit_vector;
    }

    // Build function.
    void build(void* buildParameter) {
        
        // extract build parameters
        acorn_buildParameter* param = (acorn_buildParameter*) buildParameter;
        size_t M = param->M;
        size_t gamma = param->gamma;
        size_t M_beta = param->M_beta;

        if(logger){
            logger->logging(
                "[Build] M=" + std::to_string(M)
                + ", gamma=" + std::to_string(gamma)
                + ", M_beta=" + std::to_string(M_beta),
                true
            );
        }
        else
            printf("[Build] M=%lu, gamma=%lu, M_beta=%lu\n", M, gamma, M_beta);

        // Metadata vector.
        int64_t start = timing.get_s_time();
        metadata.assign(nr_vector, 0.0);
        index = new faiss::IndexACORNFlat(dimension, M, gamma, metadata, M_beta, faiss::METRIC_INNER_PRODUCT);

        // Add vectors.
        index->add(nr_vector, vector_data);
        int64_t middle = timing.get_s_time();

        if(logger){
            logger->logging(
                "Build time = " + std::to_string(middle - start) + " s",
                true
            );
        }
        else
            printf("Build time = %ld s\n", middle - start);

        // ---- build code ----
    }

    // Query function that returns average recall.
    float query(void* searchParameter) {

        acorn_searchParameter* param = (acorn_searchParameter*) searchParameter;
        int target_id = param->target_idx;

        std::string &golden_file = param->golden_file;
        int top_k = param->top_k;
        int efSearch = param->efSearch;
        enum query_method using_method = param->method;

        if(logger){
            logger->logging(
                "[Query] golden=" + golden_file
                + ", top_k=" + std::to_string(top_k)
                + ", efSearch=" + std::to_string(efSearch),
                true
            );
        }
        else
            printf("[Query] golden=%s, top_k=%d, efSearch=%d\n", golden_file.c_str(), top_k, efSearch);

        // ---- query code ----
        // Clear historical state.
        clear_stats();
        stat_recall.set_amp(100);

        // Load ground truth.
        recall.set_metadata(golden_file);

        faiss::IDSelector *selector;

        if(using_method==QUERY_NO_FILTER)
            selector = nullptr;
        else if(using_method==QUERY_METHOD_INT_ARRAY){
            selector = new IDSelectorIntArray(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_INT_ARRAY_SLEEP){
            selector = new IDSelectorIntArraySleep(param->load_metadata_path);
            selector->set_sleep_ns(param->sleep_ns);
        }
        else if(using_method==QUERY_METHOD_MULTI_FILTER){
            selector = new IDSelectorMultiAttribute(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_REGEX){
            selector = new IDSelectorRegex(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_INT_LARGER){
            selector = new IDSelectorIntArrayLarger(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_FUZZY_FILTER){
            selector = new IDSelectorFuzzyFilter(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_FLOAT_LOWER){
            selector = new IDSelectorFloatArrayLower(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_COMPOSITE_FILTER){
            selector = new IDSelectorCompositeFiltering(param->load_metadata_path);
        }
        else{
            assert(false);
        }

        // Start search.
        float* query = query_data;
        for(int i=0; i<nr_query; i++){

            if(target_id!=-1 && target_id!=i){
                query += dimension;
                continue;
            }

            // Generate required vector values.
            std::vector<faiss::idx_t> labels(top_k * 1);
            std::vector<float> dis2(top_k * 1);
            faiss::SearchParametersACORN inner_param;
            faiss::acorn_stats.reset();

            if(using_method==QUERY_METHOD_BIT_VECTOR)
                selector = new IDSelectorBitmapLatency(nr_bytes, (uint8_t*)(query_bit_vector + i * nr_bytes));
            else if(selector){
                selector->set_filter(query_metadata[i]);
            }

            inner_param.sel = selector;
            inner_param.efSearch = efSearch;
            inner_param.check_relative_distance = true;
            inner_param.limited_mode = param->limited_mode;
            inner_param.filter_opt = param->filter_opt;
            inner_param.output_search = param->output_search;

            int64_t start_time = timing.get_us_time();
            // index->acorn.efSearch = efSearch;
            index->hybrid_search(1, query, top_k, dis2.data(), labels.data(), &inner_param);

            int64_t end_time = timing.get_us_time();

            // print_label(labels.data(), dis2.data(), 100);

            stat_latency.add(end_time - start_time);
            stat_dist_comp.add(faiss::acorn_stats.ndis);
            stat_dist_comp_first_stage.add(faiss::acorn_stats.ndis_first);
            stat_nr_filter.add(faiss::acorn_stats.nfilter);
            stat_nr_filter_first_stage.add(faiss::acorn_stats.nfilter_first);
            stat_nr_step.add(faiss::acorn_stats.nhop);
            stat_nr_step_first_stage.add(faiss::acorn_stats.nhop_first);

            query += dimension;

            double res = recall.get_recall(i, labels.data(), top_k);
            stat_recall.add(res);

            // printf("[Query] %d %f\n", i, res);

            if(using_method==QUERY_METHOD_BIT_VECTOR)
                delete selector;

        }

        if(using_method!=QUERY_METHOD_BIT_VECTOR && selector)
            delete selector;

        // Print results.
        print_results();

        return stat_recall.get_mean();

        // ---- build code ----

    }


    float batch_query(void* searchParameter) {

        acorn_searchParameter* param = (acorn_searchParameter*) searchParameter;
        int target_id = param->target_idx;

        std::string &golden_file = param->golden_file;
        int top_k = param->top_k;
        int efSearch = param->efSearch;
        enum query_method using_method = param->method;

        if(logger){
            logger->logging(
                "[Query] golden=" + golden_file
                + ", top_k=" + std::to_string(top_k)
                + ", efSearch=" + std::to_string(efSearch),
                true
            );
        }
        else
            printf("[Query] golden=%s, top_k=%d, efSearch=%d\n", golden_file.c_str(), top_k, efSearch);

        clear_stats();
        stat_recall.set_amp(100);

        recall.set_metadata(golden_file);

        faiss::IDSelector *selector;

        if(using_method==QUERY_NO_FILTER)
            selector = nullptr;
        else if(using_method==QUERY_METHOD_INT_ARRAY){
            selector = new IDSelectorIntArray(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_INT_ARRAY_SLEEP){
            selector = new IDSelectorIntArraySleep(param->load_metadata_path);
            selector->set_sleep_ns(param->sleep_ns);
        }
        else if(using_method==QUERY_METHOD_MULTI_FILTER){
            selector = new IDSelectorMultiAttribute(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_REGEX){
            selector = new IDSelectorRegex(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_INT_LARGER){
            selector = new IDSelectorIntArrayLarger(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_FUZZY_FILTER){
            selector = new IDSelectorFuzzyFilter(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_FLOAT_LOWER){
            selector = new IDSelectorFloatArrayLower(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_COMPOSITE_FILTER){
            selector = new IDSelectorCompositeFiltering(param->load_metadata_path);
        }
        else{
            assert(false);
        }


        std::vector<faiss::idx_t> labels(top_k * 1 * nr_query);
        std::vector<float> dis2(top_k * 1 * nr_query);
        faiss::SearchParametersACORN inner_param;
        faiss::acorn_stats.reset();

        selector->set_filter_batch(query_metadata);

        inner_param.sel = selector;
        inner_param.efSearch = efSearch;
        inner_param.check_relative_distance = true;
        inner_param.limited_mode = param->limited_mode;
        inner_param.filter_opt = param->filter_opt;
        inner_param.output_search = param->output_search;

        int64_t start_time = timing.get_us_time();
        index->hybrid_search(nr_query, query_data, top_k, dis2.data(), labels.data(), &inner_param);
        int64_t end_time = timing.get_us_time();

        double avg_recall = recall.get_recall_batch(nr_query, labels.data(), top_k);

        char output_content[1024];
        sprintf(output_content, "[Result] Nr_query=%u, total_time=%ld us, Tput=%.2f ops, avg_filter_first=%lu, avg_filter=%lu, avg_dist_first=%lu, avg_dist=%lu, avg_step_first=%lu, avg_step=%lu, avg_recall=%.2f %%\n", nr_query, (end_time-start_time), nr_query * 1e6 / (end_time - start_time), faiss::acorn_stats.nfilter_first / nr_query, faiss::acorn_stats.nfilter / nr_query, faiss::acorn_stats.ndis_first / nr_query, faiss::acorn_stats.ndis / nr_query, faiss::acorn_stats.nhop_first / nr_query, faiss::acorn_stats.nhop / nr_query, avg_recall * 100);

        logger->logging(std::string(output_content));

        if(selector)
            delete selector;

        return avg_recall * 100;

    }


    float auto_batch_query(void* searchParameter, float target_recall, int min_efSearch=0, int max_efSearch=5000) {

        // this means the search terminate early, quit directly
        if(max_efSearch==-1){
            logger->loggingf("[PointSearch] Nr_query=%u, total_time=%ld us, Tput=%.2f ops, avg_dist=%d, avg_filter=%d, avg_recall=%.2f %%\n", nr_query, 0, 0, 0, 0, 0);
            return 0;
        }

        acorn_searchParameter* param = (acorn_searchParameter*) searchParameter;

        std::string &golden_file = param->golden_file;
        int top_k = param->top_k;
        enum query_method using_method = param->method;

        recall.set_metadata(golden_file);

        faiss::IDSelector *selector;

        if(using_method==QUERY_NO_FILTER)
            selector = nullptr;
        else if(using_method==QUERY_METHOD_INT_ARRAY){
            selector = new IDSelectorIntArray(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_INT_ARRAY_SLEEP){
            selector = new IDSelectorIntArraySleep(param->load_metadata_path);
            selector->set_sleep_ns(param->sleep_ns);
        }
        else if(using_method==QUERY_METHOD_MULTI_FILTER){
            selector = new IDSelectorMultiAttribute(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_REGEX){
            selector = new IDSelectorRegex(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_INT_LARGER){
            selector = new IDSelectorIntArrayLarger(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_FUZZY_FILTER){
            selector = new IDSelectorFuzzyFilter(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_FLOAT_LOWER){
            selector = new IDSelectorFloatArrayLower(param->load_metadata_path);
        }
        else if(using_method==QUERY_METHOD_COMPOSITE_FILTER){
            selector = new IDSelectorCompositeFiltering(param->load_metadata_path);
        }
        else{
            assert(false);
        }


        std::vector<faiss::idx_t> labels(top_k * 1 * nr_query);
        std::vector<float> dis2(top_k * 1 * nr_query);
        faiss::SearchParametersACORN inner_param;
        faiss::acorn_stats.reset();

        selector->set_filter_batch(query_metadata);

        inner_param.sel = selector;
        inner_param.check_relative_distance = true;
        inner_param.limited_mode = param->limited_mode;
        inner_param.filter_opt = param->filter_opt;
        inner_param.output_search = param->output_search;

        // search for a suitable efSearch
        int suitable_efSearch = min_efSearch;
        float suitable_recall = 0;
        int64_t suitable_time = 0;

        // the boundary between min_efSearch and max_efSearch may be too large, we use the binary search instead

        float recall_min_pos = -1;
        int64_t time_min_pos = 0;
        float recall_max_pos = -1;
        int64_t time_max_pos = 0;
        int ndist_min = 0, nfilter_min = 0;
        int ndist_max = 0, nfilter_max = 0;

        while(min_efSearch + 1 < max_efSearch){

            int efSearch = (min_efSearch + max_efSearch) / 2;

            inner_param.efSearch = efSearch;
            faiss::acorn_stats.reset();
            int64_t start_time = timing.get_us_time();
            index->hybrid_search(nr_query, query_data, top_k, dis2.data(), labels.data(), &inner_param);
            int64_t end_time = timing.get_us_time();
            double avg_recall = recall.get_recall_batch(nr_query, labels.data(), top_k) * 100;

            if(avg_recall <= target_recall){
                min_efSearch = efSearch;
                recall_min_pos = avg_recall;
                time_min_pos = end_time - start_time;
                ndist_min = faiss::acorn_stats.ndis;
                nfilter_min = faiss::acorn_stats.nfilter;
            }
            else{
                max_efSearch = efSearch;
                recall_max_pos = avg_recall;
                time_max_pos = end_time - start_time;
                ndist_max = faiss::acorn_stats.ndis;
                nfilter_max = faiss::acorn_stats.nfilter;
            }

        }

        // we cannot let left equal right
        if(min_efSearch==max_efSearch)
            max_efSearch++;

        // one of them may not be calculated, do it
        if(recall_min_pos==-1){
            inner_param.efSearch = min_efSearch;
            faiss::acorn_stats.reset();
            int64_t start_time = timing.get_us_time();
            index->hybrid_search(nr_query, query_data, top_k, dis2.data(), labels.data(), &inner_param);
            int64_t end_time = timing.get_us_time();
            recall_min_pos = recall.get_recall_batch(nr_query, labels.data(), top_k) * 100;
            time_min_pos = end_time - start_time;
            ndist_min = faiss::acorn_stats.ndis;
            nfilter_min = faiss::acorn_stats.nfilter;
        }
        if(recall_max_pos==-1){
            inner_param.efSearch = max_efSearch;
            faiss::acorn_stats.reset();
            int64_t start_time = timing.get_us_time();
            index->hybrid_search(nr_query, query_data, top_k, dis2.data(), labels.data(), &inner_param);
            int64_t end_time = timing.get_us_time();
            recall_max_pos = recall.get_recall_batch(nr_query, labels.data(), top_k) * 100;
            time_max_pos = end_time - start_time;                ndist_max = faiss::acorn_stats.ndis;
            nfilter_max = faiss::acorn_stats.nfilter;
        }

        assert(recall_min_pos<=target_recall && recall_max_pos>=target_recall);

        if(!(recall_min_pos<=target_recall && recall_max_pos>=target_recall)){
            printf("min=%f, max=%f, target=%f, minEfSearch=%d, maxEfSearch=%d\n", recall_min_pos, recall_max_pos, target_recall, min_efSearch, max_efSearch);
            assert(false);
        }

        int ndist = 0, nfilter = 0;
        if(target_recall - recall_min_pos <= recall_max_pos - target_recall){
            suitable_efSearch = min_efSearch;
            suitable_recall = recall_min_pos;
            suitable_time = time_min_pos;
            ndist = ndist_min;
            nfilter = nfilter_min;
        }
        else{
            suitable_efSearch = max_efSearch;
            suitable_recall = recall_max_pos;
            suitable_time = time_max_pos;
            ndist = ndist_max;
            nfilter = nfilter_max;
        }


        logger->loggingf("[AutoDeciding] target_recall=%f efSearch=%d true_recall=%f\n", target_recall, suitable_efSearch, suitable_recall);

        float tput = suitable_time ? nr_query * 1e6 / suitable_time : 0;
        logger->loggingf("[PointSearch] Nr_query=%u, total_time=%ld us, Tput=%.2f ops, avg_filter=%d, avg_dist=%d, avg_recall=%.2f %%\n", nr_query, suitable_time, tput, nfilter / nr_query, ndist / nr_query, suitable_recall);

        if(selector)
            delete selector;

        return suitable_recall;

    }


    // clear function; call it after each build test.
    void clear() {
        if(index)
            delete index;
        index = nullptr;
        metadata.clear();
    }

    void save_persist(char* persist_path){

        if(!index){
            printf("Error: empty index!\n");
            return;
        }
        faiss::write_index(index, persist_path);
    }

    void load_persist(char* persist_path){

        index = dynamic_cast<faiss::IndexACORNFlat*>(faiss::read_index(persist_path));

    }

    int get_nr_query(){
        return this->nr_query;
    }

};

#endif