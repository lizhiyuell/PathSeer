#ifndef __VBASE_HPP__
#define __VBASE_HPP__

#include "bencher.hpp"
#include <hnswalg.h>
#include <iostream>
#include <string>
#include <cassert>
#include <stdio.h>

typedef struct{
    size_t M;
} vbase_buildParameter;

typedef struct{
    std::string golden_file;
    int top_k;
    int efSearch;
} vbase_searchParameter;

class CustomFilterFunctor: public hnswlib::BaseFilterFunctor {

 public:
    char* id_arr;
    explicit CustomFilterFunctor(char* id_selector) {
        id_arr = id_selector;
    }

    bool operator()(hnswlib::labeltype id) {
        return id_arr[id]==1;
    }
};

class Vbase_bencher : public Bencher {
private:

    Recall<hnswlib::labeltype> recall;

    // Search index.
    hnswlib::HierarchicalNSW<float> *index {nullptr};
    hnswlib::SpaceInterface<float> *spaceInterface {nullptr};

    // Variable definitions.
    float *vector_data, *query_data;
    char *query_bit_vector;
    uint32_t nr_vector, dimension, nr_query;

    std::string base_dir;

public:
    Vbase_bencher(std::string base_dir, std::string metadata_name) {
        if(logger)
            logger->logging("--- Testing Vbase_bencher of " + base_dir + ": " + metadata_name + " ---\n", true);
        else
            printf("--- Testing Vbase_bencher of %s:%s ---\n", base_dir.c_str(), metadata_name.c_str());
        // Hybrid search: Load vector data and query data, as well as metadata, in bv form
        load_ann_data_bv(
            base_dir, metadata_name,
            &vector_data, &nr_vector, &dimension,
            &query_data, &nr_query, &query_bit_vector,
            true
        );
    }

    Vbase_bencher(std::string base_dir) {
        if(logger)
            logger->logging("--- Testing Vbase_bencher of " + base_dir + " ---\n", true);
        else
            printf("--- Testing Vbase_bencher of %s ---\n", base_dir.c_str());
        // Hybrid search: Load vector data and query data, as well as metadata, in bv form
        this->base_dir = base_dir;
        load_ann_data(
            base_dir,
            &vector_data, &nr_vector, &dimension,
            &query_data, &nr_query,
            true
        );
    }

    void set_metadata(std::string metadata_class){
        if(query_bit_vector)
            delete query_bit_vector;
        uint32_t this_nr_query;
        load_ann_metadata_bv(this->base_dir, nr_vector, metadata_class, &this_nr_query, &query_bit_vector);
        assert(nr_query==this_nr_query);
    }

    // Destructor.
    ~Vbase_bencher() {
        delete vector_data;
        delete query_data;
        delete query_bit_vector;
    }

    // Build function.
    void build(void* buildParameter) {
        
        // extract build parameters
        vbase_buildParameter* param = (vbase_buildParameter*) buildParameter;
        size_t M = param->M;

        if(logger){
            logger->logging(
                "[Build] M=" + std::to_string(M),
                true
            );
        }
        else
            printf("[Build] M=%lu\n", M);

        // Metadata vector.
        int64_t start = timing.get_s_time();
        spaceInterface = new hnswlib::InnerProductSpace(dimension);

        index = new hnswlib::HierarchicalNSW<float>(
            spaceInterface,
            nr_vector,
            M);

        // HNSWLIB requires IDs; use sequential IDs.
        hnswlib::labeltype *id_list = new hnswlib::labeltype[nr_vector];
        for(int i=0; i<nr_vector; i++)
            id_list[i] = i;

        // Add vectors.
        index->addPoints(nr_vector, vector_data, id_list);
        int64_t middle = timing.get_s_time();

        if(logger){
            logger->logging(
                "Build time = " + std::to_string(middle - start) + " s",
                true
            );
        }
        else
            printf("Build time = %ld s\n", middle - start);

        delete id_list;

        // ---- build code ----
    }

    // Query function that returns average recall.
    float query(void* searchParameter) {

        vbase_searchParameter* param = (vbase_searchParameter*) searchParameter;

        std::string &golden_file = param->golden_file;
        int top_k = param->top_k;
        int efSearch = param->efSearch;

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
        stat_latency.clear();
        stat_recall.clear();
        stat_recall.set_amp(100);
        stat_dist_comp.clear();

        // Load ground truth.
        recall.set_metadata(golden_file);

        // Start search.
        float* query = query_data;
        for(int i=0; i<nr_query; i++){

            // Generate required vector values.
            std::vector<hnswlib::labeltype> labels(top_k * 1);
            std::vector<float> dis2(top_k * 1);

            // Metadata filter structure.
            CustomFilterFunctor filter_function(query_bit_vector + i * nr_vector);

            index->setEf(efSearch);

            int64_t start_time = timing.get_us_time();

            // std::priority_queue<std::pair<float, hnswlib::labeltype >> result = index->searchKnn(query, top_k, &filter_function);
            index->searchKnnParallelReturnFlat(1, query, top_k, labels.data(), dis2.data(), &filter_function);

            int64_t end_time = timing.get_us_time();

            // print_label(labels.data(), dis2.data(), 100);

            stat_latency.add(end_time - start_time);

            query += dimension;

            // Calculate recall.
            double res = recall.get_recall(i, labels.data(), top_k);
            stat_recall.add(res);
            stat_dist_comp.add(index->search_counter[5]);

            // print_recall(res);

            // break;
        }

        // Print results.

        if(logger){
            logger->logging(
                "MeanTime(us): " + std::to_string(stat_latency.get_mean())
                + ", StdTime(us): " + std::to_string(stat_latency.get_std())
                + ", MedianTime(us): " + std::to_string(stat_latency.get_percentile(50))
                + ", P90Time(us): " + std::to_string(stat_latency.get_percentile(90))
                + ", P95Time(us): " + std::to_string(stat_latency.get_percentile(95))
                + ", P99Time(us): " + std::to_string(stat_latency.get_percentile(99))
            );
        }
        else
            printf("Mean time %d us, std time %d us, Median time %d us, P90 time %d us, P95 time %d us, P99 time %d us\n", stat_latency.get_mean(), stat_latency.get_std(), stat_latency.get_percentile(50), stat_latency.get_percentile(90), stat_latency.get_percentile(95), stat_latency.get_percentile(99));

        if(logger){
            logger->logging(
                "MeanRecall: " + std::to_string(stat_recall.get_mean())
                + ", StdRecall: " + std::to_string(stat_recall.get_std())
                + ", MedianRecall: " + std::to_string(stat_recall.get_percentile(50))
                + ", P1Recall: " + std::to_string(stat_recall.get_percentile(1))
                + ", P5Recall: " + std::to_string(stat_recall.get_percentile(5))
                + ", P10Recall: " + std::to_string(stat_recall.get_percentile(10))
            );
        }
        else
            printf("Mean recall %.2f, std recall %.2f, Median recall %.2f, P1 recall %.2f, P5 recall %.2f, P10 recall %.2f\n", stat_recall.get_mean(), stat_recall.get_std(), stat_recall.get_percentile(50), stat_recall.get_percentile(1), stat_recall.get_percentile(5), stat_recall.get_percentile(10));

        if(logger){
            logger->logging(
                "MeanDistComp: " + std::to_string(stat_dist_comp.get_mean())
                + ", StdDistComp: " + std::to_string(stat_dist_comp.get_std())
                + ", MedianDistComp: " + std::to_string(stat_dist_comp.get_percentile(50))
                + ", P90DistComp: " + std::to_string(stat_dist_comp.get_percentile(90))
                + ", P95DistComp: " + std::to_string(stat_dist_comp.get_percentile(95))
                + ", P99DistComp: " + std::to_string(stat_dist_comp.get_percentile(99))
            );
        }
        else
            printf("Mean DistComp %d, std DistComp %d, Median DistComp %d, P1 DistComp %d, P5 DistComp %d, P10 DistComp %d\n", stat_dist_comp.get_mean(), stat_dist_comp.get_std(), stat_dist_comp.get_percentile(50), stat_dist_comp.get_percentile(1), stat_dist_comp.get_percentile(5), stat_dist_comp.get_percentile(10));

        fflush(stdout);

        return stat_recall.get_mean();

        // ---- build code ----

    }

    // clear function; call it after each build test.
    void clear() {
        if(index)
            delete index;
        index = nullptr;
        if(spaceInterface)
            delete spaceInterface;
        spaceInterface = nullptr;
    }

    void save_persist(char* persist_path){

        if(!index){
            printf("Error: empty index!\n");
            return;
        }
        index->saveIndex(persist_path);

    }

    void load_persist(char* persist_path){

        spaceInterface = new hnswlib::InnerProductSpace(dimension);
        index = new hnswlib::HierarchicalNSW<float>(spaceInterface, std::string(persist_path));

    }

};

#endif