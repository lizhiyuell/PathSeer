#ifndef __IVFFLAT_HPP__
#define __IVFFLAT_HPP__

#include "bencher.hpp"
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/impl/platform_macros.h>
#include <iostream>
#include <string>
#include <cassert>
#include <stdio.h>

typedef struct{
    int nlist;
} filtered_ivf_buildParameter;

typedef struct{
    std::string golden_file;
    int top_k;
    int cluster;//=nlist
    int nprobe;
    enum query_method method = QUERY_NO_FILTER;
    int sleep_ns = 0;
    int target_idx = -1;
    std::string load_metadata_path; // path for dataset metadata
} filtered_ivf_searchParameter;

class FILTERED_IVF_bencher : public Bencher {
private:
    faiss::IndexIVFFlat *index {nullptr};

    float *vector_data {nullptr}, *query_data {nullptr};
    char *query_bit_vector {nullptr};
    uint32_t nr_vector, dimension, nr_query, nr_bytes;

    std::string base_dir;

public:
    FILTERED_IVF_bencher(std::string base_dir, std::string metadata_name) {
        if(logger)
            logger->logging("--- Testing FILTERED_IVF_bencher of " + base_dir + ": " + metadata_name + " ---\n", true);
        else
            printf("--- Testing FILTERED_IVF_bencher of %s:%s ---\n", base_dir.c_str(), metadata_name.c_str());
        // Hybrid search: Load vector data and query data, as well as metadata, in bv form
        load_ann_data_bv_tight(
            base_dir, metadata_name,
            &vector_data, &nr_vector, &dimension,
            &query_data, &nr_query, &nr_bytes, &query_bit_vector,
            true
        );
    }

    FILTERED_IVF_bencher(std::string base_dir) {
        if(logger)
            logger->logging("--- Testing FILTERED_IVF_bencher of " + base_dir + " ---\n", true);
        else
            printf("--- Testing FILTERED_IVF_bencher of %s ---\n", base_dir.c_str());
        this->base_dir = base_dir;
        load_ann_data(
            base_dir,
            &vector_data, &nr_vector, &dimension,
            &query_data, &nr_query,
            true
        );
    }

    // load metadata with bit vector
    void set_metadata_bv(std::string metadata_class){
        if(query_bit_vector)
            delete query_bit_vector;
        uint32_t this_nr_query;
        load_ann_metadata_bv_tight(this->base_dir, metadata_class, &this_nr_query, &nr_bytes, &query_bit_vector);
        assert(nr_query==this_nr_query);
    }

    ~FILTERED_IVF_bencher() {
        delete vector_data;
        delete query_data;
        delete query_bit_vector;
    }

    void build(void* buildParameter) {
        // extract build parameters
        filtered_ivf_buildParameter* param = (filtered_ivf_buildParameter*) buildParameter;
        size_t nlist = param->nlist;
        if(logger){
            logger->logging(
                "[Build] nlist=" + std::to_string(nlist),
                true
            );
        }
        else
            printf("[Build] nlist=%lu\n", nlist);

        int64_t start = timing.get_s_time();
        // faiss::IndexFlatIP quantizer(dimension);
        // index = new faiss::IndexIVFFlat(&quantizer, dimension, nlist, faiss::METRIC_INNER_PRODUCT);
        // error：Segmentation fault (core dumped)
        faiss::IndexFlatIP* quantizer = new faiss::IndexFlatIP(dimension);
        index = new faiss::IndexIVFFlat(quantizer, dimension, nlist, faiss::METRIC_INNER_PRODUCT);

        // IVF index needs to be trained first
        if (!index->is_trained) {
            index->train(nr_vector, vector_data);
        }
        // add vectors
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

    }

    // query with bitvector
    float query(void* searchParameter) {
        filtered_ivf_searchParameter* param = (filtered_ivf_searchParameter*) searchParameter;
        int target_id = param->target_idx;

        std::string &golden_file = param->golden_file;
        int top_k = param->top_k;
        int nprobe = param->nprobe;
        index->nprobe = param->nprobe;
        enum query_method using_method = param->method;

        if(logger){
            logger->logging(
                "[Query] golden=" + golden_file
                + ", nr_query=" + std::to_string(nr_query)
                + ", top_k=" + std::to_string(top_k)
                + ", nprobe=" + std::to_string(nprobe),
                true
            );
        }
        else
            printf("[Query] golden=%s, top_k=%d, nprobe=%d\n", golden_file.c_str(), top_k, nprobe);

        // Clear historical status
        clear_stats();
        stat_recall.set_amp(100);

        // load ground truth
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

        // Starting retrieval
        float* query = query_data;
        for(int i=0; i<nr_query; i++){
            if(target_id!=-1 && target_id!=i){
                query += dimension;
                continue;
            }

            std::vector<faiss::idx_t> labels(top_k * 1);
            std::vector<float> dis2(top_k * 1);
            faiss::IVFSearchParameters search_param;
            faiss::indexIVF_stats.reset();

            if(using_method==QUERY_METHOD_BIT_VECTOR)
                selector = new IDSelectorBitmapLatency(nr_bytes, (uint8_t*)(query_bit_vector + i * nr_bytes));
            else if(selector){
                selector->set_filter(query_metadata[i]);
            }

            // search_param.check_relative_distance = true;
            search_param.nprobe = param->nprobe;
            search_param.sel = selector;//index.h 67line

            int64_t start_time = timing.get_us_time();
            index->search(1, query, top_k, dis2.data(), labels.data(), &search_param);
            int64_t end_time = timing.get_us_time();

            stat_latency.add(end_time - start_time);
            stat_dist_comp_first_stage.add(faiss::indexIVF_stats.nrealcomp);//The number of distance calculations in all inverted lists
            stat_dist_comp.add(faiss::indexIVF_stats.nrealcomp+param->cluster);//The number of distance calculations in the entire query
            stat_nr_step_first_stage.add(faiss::indexIVF_stats.quantization_time*1000);//Time to search for cluster centers(us)
            stat_nr_step.add(faiss::indexIVF_stats.search_time*1000);//The total time of the query process(us)
            stat_nr_filter_first_stage.add(faiss::indexIVF_stats.nheap_updates);//The number of updates for the heap
            stat_nr_filter.add(faiss::indexIVF_stats.nfilter);//The number of filtering operations
            
            query += dimension;

            // Calculate Recall
            double res = recall.get_recall(i, labels.data(), top_k);
            stat_recall.add(res);

            if(using_method==QUERY_METHOD_BIT_VECTOR)
                delete selector;
        }

        if(using_method!=QUERY_METHOD_BIT_VECTOR && selector)
            delete selector;

        print_results();//bencher.hpp

        return stat_recall.get_mean();
    }

    float batch_query(void* searchParameter) {
        filtered_ivf_searchParameter* param = (filtered_ivf_searchParameter*) searchParameter;
        int target_id = param->target_idx;

        std::string &golden_file = param->golden_file;
        int top_k = param->top_k;
        int nprobe = param->nprobe;
        index->nprobe = param->nprobe;
        enum query_method using_method = param->method;

        if(logger){
            logger->logging(
                "[Query] golden=" + golden_file
                + ", nr_query=" + std::to_string(nr_query)
                + ", top_k=" + std::to_string(top_k)
                + ", nprobe=" + std::to_string(nprobe),
                true
            );
        }
        else
            printf("[Query] golden=%s, top_k=%d, nprobe=%d\n", golden_file.c_str(), top_k, nprobe);

        // Clear historical status
        clear_stats();
        stat_recall.set_amp(100);

        // load ground truth
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
        faiss::IVFSearchParameters search_param;
        faiss::indexIVF_stats.reset();

        selector->set_filter_batch(query_metadata);

        // search_param.check_relative_distance = true;
        search_param.nprobe = param->nprobe;
        search_param.sel = selector;

        int64_t start_time = timing.get_us_time();
        index->search(nr_query, query_data, top_k, dis2.data(), labels.data(), &search_param);
        int64_t end_time = timing.get_us_time();

        double avg_recall = recall.get_recall_batch(nr_query, labels.data(), top_k);

        char output_content[1024];
        sprintf(output_content, "[Result] Nr_query=%u, total_time=%ld us, Tput=%.2f ops, avg_filter=%lu, avg_dist=%lu, avg_recall=%.2f %%\n", nr_query, (end_time-start_time), nr_query * 1e6 / (end_time - start_time), (faiss::indexIVF_stats.nrealcomp+param->cluster) / nr_query, faiss::indexIVF_stats.nfilter / nr_query, avg_recall * 100);

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

        filtered_ivf_searchParameter* param = (filtered_ivf_searchParameter*) searchParameter;

        std::string &golden_file = param->golden_file;
        int top_k = param->top_k;
        enum query_method using_method = param->method;

        // load ground truth
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
        faiss::IVFSearchParameters search_param;
        faiss::indexIVF_stats.reset();

        selector->set_filter_batch(query_metadata);

        search_param.sel = selector;


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

            search_param.nprobe = efSearch;
            faiss::indexIVF_stats.reset();
            int64_t start_time = timing.get_us_time();
            index->search(nr_query, query_data, top_k, dis2.data(), labels.data(), &search_param);
            int64_t end_time = timing.get_us_time();
            double avg_recall = recall.get_recall_batch(nr_query, labels.data(), top_k) * 100;

            if(avg_recall <= target_recall){
                min_efSearch = efSearch;
                recall_min_pos = avg_recall;
                time_min_pos = end_time - start_time;
                ndist_min = faiss::indexIVF_stats.nrealcomp+param->cluster;
                nfilter_min = faiss::indexIVF_stats.nfilter;
            }
            else{
                max_efSearch = efSearch;
                recall_max_pos = avg_recall;
                time_max_pos = end_time - start_time;
                ndist_max = faiss::indexIVF_stats.nrealcomp+param->cluster;
                nfilter_max = faiss::indexIVF_stats.nfilter;
            }

        }

        // we cannot let left equal right
        if(min_efSearch==max_efSearch)
            max_efSearch++;

        // one of them may not be calculated, do it
        if(recall_min_pos==-1){
            search_param.nprobe = min_efSearch;
            faiss::indexIVF_stats.reset();
            int64_t start_time = timing.get_us_time();
            index->search(nr_query, query_data, top_k, dis2.data(), labels.data(), &search_param);
            int64_t end_time = timing.get_us_time();
            recall_min_pos = recall.get_recall_batch(nr_query, labels.data(), top_k) * 100;
            time_min_pos = end_time - start_time;
            ndist_min = faiss::indexIVF_stats.nrealcomp+param->cluster;
            nfilter_min = faiss::indexIVF_stats.nfilter;
        }
        if(recall_max_pos==-1){
            search_param.nprobe = max_efSearch;
            faiss::indexIVF_stats.reset();
            int64_t start_time = timing.get_us_time();
            index->search(nr_query, query_data, top_k, dis2.data(), labels.data(), &search_param);
            int64_t end_time = timing.get_us_time();
            recall_max_pos = recall.get_recall_batch(nr_query, labels.data(), top_k) * 100;
            time_max_pos = end_time - start_time;
            ndist_max = faiss::indexIVF_stats.nrealcomp+param->cluster;
            nfilter_max = faiss::indexIVF_stats.nfilter;
        }

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


    // after build or query need clear()
    void clear() {
        if(index)
            delete index;
        index = nullptr;
    }

    void save_persist(char* persist_path){

        if(!index){
            printf("Error: empty index!\n");
            return;
        }
        faiss::write_index(index, persist_path);
    }

    void load_persist(char* persist_path){

        if(index)
            delete index;

        index = dynamic_cast<faiss::IndexIVFFlat*>(faiss::read_index(persist_path));

        // set the parallel mode
        index->parallel_mode = 3;

    }

    int get_nr_query(){
        return this->nr_query;
    }

};

#endif