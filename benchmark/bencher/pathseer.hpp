#ifndef __PathSeer_HPP__
#define __PathSeer_HPP__

#include "bencher.hpp"
#include <faiss/IndexFlat.h>
#include <faiss/IndexPathSeer.h>
#include <faiss/index_io.h>
#include <iostream>
#include <string>
#include <cassert>
#include <stdio.h>
#include "../utils/logger.hpp"
#include "../utils/normalize.hpp"
#include <random>
#include <unordered_set>
#include <algorithm>
#include <climits>
#include <queue>
#include <cfloat>
#include <sstream>
#include <functional>
#include <utility>
#include <immintrin.h>
#include <sched.h>
#include <cmath>
#include <thread>
#include <chrono>
#include <atomic>

typedef struct{
    int M;
    int M_exp;
    int efConstruction = 0;
    int efConstructionExp = 0;
    bool use_only_expanding = false;
    bool use_optimized_building = true;
} pathseer_buildParameter;

typedef struct{
    std::string golden_file;
    int top_k;
    int efSearch;
    int M_exp_search;
    // the overhead added to filter
    int sleep_ns = 0;
    enum query_method method = QUERY_NO_FILTER;
    int target_idx = -1; // maybe we only want to query a certain idx
    bool profile_virtual_overhead = false; // whether print the result to the virtual overhead
    bool print_virtual_overhead = false;
    int virtual_overhead = 0; // we may set the virtual overhead by ourselves
    // the coefficient for virtual overhead
    float coef_a = 0;
    float coef_b = 0;
    bool use_mff_only = false; // whether search with MFF only method
    bool use_two_hop_search = false; // whether use the two-hop search, this has lower priority than the "use_mff_only" method
    bool use_dfn_only_search = false;
    std::string load_metadata_path; // path for dataset metadata
} pathseer_searchParameter;

typedef struct VOP{

    float eval_time_inv = 0;
    int virtual_overhead_ns = 0;

    VOP(float inv_time, int overhead) {
        eval_time_inv = inv_time;
        virtual_overhead_ns = overhead;
    }

    bool operator<(const VOP& other) const {
        return eval_time_inv > other.eval_time_inv;
    }

} virtual_overhead_pair;

// the class used for golden generation
struct Pair {
    float score;
    int id;
};

class TopKHeapArray {
public:
    explicit TopKHeapArray(size_t k) : cap_(k), n_(0), a_(k) {}

    void add(float score, int id) {
        if (n_ < cap_) {
            a_[n_] = {score, id};
            siftUp(n_);
            ++n_;
            return;
        }
        if (score <= a_[0].score) return;
        a_[0] = {score, id};
        siftDown(0);
    }

    const Pair& top() const {
        if (n_ == 0) throw std::out_of_range("heap is empty");
        return a_[0];
    }

    Pair pop() {
        if (n_ == 0) throw std::out_of_range("heap is empty");
        Pair ret = a_[0];
        a_[0] = a_[n_ - 1];
        --n_;
        if (n_ > 0) siftDown(0);
        return ret;
    }

    size_t size() const { return n_; }
    size_t capacity() const { return cap_; }
    bool empty() const { return n_ == 0; }
    bool full() const { return n_ == cap_; }

private:
    // Min-heap: parent <= child by score.
    static bool lessEq(const Pair& x, const Pair& y) {
        // Customize NaN handling here if needed; otherwise use default floating-point comparison.
        return x.score <= y.score;
    }

    void siftUp(size_t i) {
        while (i > 0) {
            size_t p = (i - 1) / 2;
            if (lessEq(a_[p], a_[i])) break;
            std::swap(a_[p], a_[i]);
            i = p;
        }
    }

    void siftDown(size_t i) {
        while (true) {
            size_t l = 2 * i + 1;
            size_t r = 2 * i + 2;
            size_t smallest = i;
            if (l < n_ && !lessEq(a_[smallest], a_[l])) smallest = l;
            if (r < n_ && !lessEq(a_[smallest], a_[r])) smallest = r;
            if (smallest == i) break;
            std::swap(a_[i], a_[smallest]);
            i = smallest;
        }
    }

    size_t cap_;
    size_t n_;
    std::vector<Pair> a_;
};

// dist comp functions
static inline __m128 _masked_read(int d, const float* x) {
    assert(0 <= d && d < 4);
    ALIGNED(16) float buf[4] = {0, 0, 0, 0};
    switch (d) {
        case 3:
            buf[2] = x[2];
        case 2:
            buf[1] = x[1];
        case 1:
            buf[0] = x[0];
    }
    return _mm_load_ps(buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}

float fvec_inner_product_avx2(const float* x, const float* y, size_t d){
    __m256 msum1 = _mm256_setzero_ps();

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(mx, my));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = _masked_read(d, x);
        __m128 my = _masked_read(d, y);
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
    }

    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    return _mm_cvtss_f32(msum2);
}

// // helper function to evalute one round

// typedef struct{
//     faiss::IndexPathSeerFlat *index;
//     int nr_artificial_vector;
//     float *artificial_vector_array;
//     int top_k;
//     float *dist_arr;
//     int64_t *retrieved_label_arr;
//     faiss::SearchParametersPathSeer *inner_param;
// }bench_args;

// void* bench_one_round(void* arg){

//     bench_args *args = (bench_args*) arg;

//     args->index->search(args->nr_artificial_vector, args->artificial_vector_array, args->top_k, args->dist_arr, args->retrieved_label_arr, args->inner_param);

//     return nullptr;

// }

class PathSeer_bencher : public Bencher {
private:
    // Search index.
    faiss::IndexPathSeerFlat *index {nullptr};

    // Variable definitions.
    float *vector_data {nullptr}, *query_data {nullptr};
    char *query_bit_vector {nullptr};
    uint32_t nr_vector, dimension, nr_query, nr_bytes;

    std::vector<int> metadata;

    std::string base_dir;

public:

    PathSeer_bencher(std::string base_dir, std::string metadata_name) {
        if(logger)
            logger->logging("--- Testing multibench_PathSeer of " + base_dir + ": " + metadata_name + " ---\n", true);
        else
            printf("--- Testing multibench_PathSeer of %s:%s ---\n", base_dir.c_str(), metadata_name.c_str());
        // Hybrid search: Load vector data and query data, as well as metadata, in bv form
        load_ann_data_bv_tight(
            base_dir, metadata_name,
            &vector_data, &nr_vector, &dimension,
            &query_data, &nr_query, &nr_bytes, &query_bit_vector,
            true
        );
    }

    // load pure data without metadata involved
    PathSeer_bencher(std::string base_dir) {
        if(logger)
            logger->logging("--- Testing multibench_PathSeer of " + base_dir + " ---\n", true);
        else
            printf("--- Testing multibench_PathSeer of %s ---\n", base_dir.c_str());
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
    ~PathSeer_bencher() {
        delete vector_data;
        delete query_data;
        delete query_bit_vector;
    }

    // Build function.
    void build(void* buildParameter) {
        
        // extract build parameters
        pathseer_buildParameter* param = (pathseer_buildParameter*) buildParameter;
        int M = param->M;
        int M_exp = param->M_exp;
        bool only_expanding = param->use_only_expanding;
        bool opt_building = param->use_optimized_building;

        if(logger){
            logger->logging(
                "[Build] M=" + std::to_string(M)
                + ", M_exp=" + std::to_string(M_exp),
                true
            );
        }
        else
            printf("[Build] M=%d, M_exp=%d\n", M, M_exp);

        int64_t start = timing.get_s_time();
        index = new faiss::IndexPathSeerFlat(dimension, M, M_exp, faiss::METRIC_INNER_PRODUCT);

        if(param->efConstruction)
            index->pathseer.efConstruction = param->efConstruction;
        if(param->efConstructionExp)
            index->pathseer.efConstructionExp = param->efConstructionExp;
        index->set_using_only_expanding_idx(only_expanding);
        index->set_using_optimized_building(opt_building);

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

        pathseer_searchParameter* param = (pathseer_searchParameter*) searchParameter;
        int target_id = param->target_idx;

        std::string &golden_file = param->golden_file;
        int top_k = param->top_k;
        int efSearch = param->efSearch;
        enum query_method using_method = param->method;

        // printf("[efSearch] %d\n", efSearch);

        if(logger){
            logger->logging(
                "[Query] golden=" + golden_file
                + ", top_k=" + std::to_string(top_k)
                + ", efSearch=" + std::to_string(efSearch) + ", M_exp_search=" + std::to_string(param->M_exp_search),
                true
            );
        }
        else
            printf("[Query] golden=%s, top_k=%d, efSearch=%d, M_exp_search=%d\n", golden_file.c_str(), top_k, efSearch, param->M_exp_search);

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
            faiss::SearchParametersPathSeer inner_param;
            faiss::pathseer_stats.reset();

            if(using_method==QUERY_METHOD_BIT_VECTOR)
                selector = new IDSelectorBitmapLatency(nr_bytes, (uint8_t*)(query_bit_vector + i * nr_bytes));
            else if(selector){
                selector->set_filter(query_metadata[i]);
            }

            inner_param.sel = selector;
            inner_param.efSearch = efSearch;
            inner_param.check_relative_distance = true;
            inner_param.M_exp_search = param->M_exp_search;
            inner_param.virtual_overhead = param->virtual_overhead;
            inner_param.coef_a = param->coef_a;
            inner_param.coef_b = param->coef_b;
            if(param->use_mff_only)
                inner_param.only_perform_mff = param->use_mff_only;
            else if(param->use_two_hop_search)
                inner_param.two_hop_search = param->use_two_hop_search;
            else if(param->use_dfn_only_search)
                inner_param.only_perform_dfn = param->use_dfn_only_search;

            // printf("[Query] %d\n", i);

            int64_t start_time = timing.get_us_time();
            index->search(1, query, top_k, dis2.data(), labels.data(), &inner_param);

            int64_t end_time = timing.get_us_time();

            stat_latency.add(end_time - start_time);
            stat_dist_comp.add(faiss::pathseer_stats.ndis);
            stat_dist_comp_first_stage.add(faiss::pathseer_stats.ndis_first);
            stat_nr_filter.add(faiss::pathseer_stats.nfilter);
            stat_nr_filter_first_stage.add(faiss::pathseer_stats.nfilter_first);
            stat_nr_step.add(faiss::pathseer_stats.nhop);
            stat_nr_step_first_stage.add(faiss::pathseer_stats.nhop_first);

            query += dimension;

            double res = recall.get_recall(i, labels.data(), top_k);
            stat_recall.add(res);

            // printf("[summary] Query=%d recall=%f dist=%d filter=%d latency=%d\n", i, res, faiss::pathseer_stats.ndis, faiss::pathseer_stats.nfilter, end_time - start_time);

            // printf("[Query] %d %f\n", i, res);

            if(using_method==QUERY_METHOD_BIT_VECTOR)
                delete selector;

        }

        if(using_method!=QUERY_METHOD_BIT_VECTOR && selector)
            delete selector;

        // printf("[Query]\n");
        // for(int i=0; i<stat_latency.size(); i++){
        //     printf("Request %d: %d us\n", i, stat_latency[i]);
        // }

        // Print results.
        print_results();

        // printf("[Mean latency] %d us", stat_latency.get_mean());

        return stat_recall.get_mean();

        // ---- build code ----

    }

    // a dedicated profile function, with less dependency
    void profile_ops_cost(enum query_method using_method, int sleep_ns, bool output_overhead=false){

        faiss::IDSelector *selector;

        if(using_method==QUERY_NO_FILTER)
            selector = nullptr;
        else if(using_method==QUERY_METHOD_INT_ARRAY){
            selector = new IDSelectorIntArray(this->base_dir + "/load/load.meta");
        }
        else if(using_method==QUERY_METHOD_INT_ARRAY_SLEEP){
            selector = new IDSelectorIntArraySleep(this->base_dir + "/load/load.meta");
            selector->set_sleep_ns(sleep_ns);
        }
        else if(using_method==QUERY_METHOD_MULTI_FILTER){
            selector = new IDSelectorMultiAttribute(this->base_dir + "/load/load.meta");
        }
        else if(using_method==QUERY_METHOD_REGEX){
            selector = new IDSelectorRegex(this->base_dir + "/load/load.meta");
        }
        else if(using_method==QUERY_METHOD_INT_LARGER){
            selector = new IDSelectorIntArrayLarger(this->base_dir + "/load/load.meta");
        }
        else if(using_method==QUERY_METHOD_FUZZY_FILTER){
            selector = new IDSelectorFuzzyFilter(this->base_dir + "/load/load.meta");
        }
        else if(using_method==QUERY_METHOD_FLOAT_LOWER){
            selector = new IDSelectorFloatArrayLower(this->base_dir + "/load/load.meta");
        }
        else if(using_method==QUERY_METHOD_COMPOSITE_FILTER){
            selector = new IDSelectorCompositeFiltering(this->base_dir + "/load/load.meta");
        }
        else{
            assert(false);
        }

        selector->set_filter_batch(query_metadata);

        index->profile_filter_cost(selector);
        index->profile_dist_comp_cost();

        if(output_overhead)
            printf("[Overhead] %lld %lld\n", index->get_profiled_dist_comp_ns(), index->get_profiled_filter_ns());

    }

    float batch_query(void* searchParameter) {

        pathseer_searchParameter* param = (pathseer_searchParameter*) searchParameter;
        int target_id = param->target_idx;

        std::string &golden_file = param->golden_file;
        int top_k = param->top_k;
        int efSearch = param->efSearch;
        enum query_method using_method = param->method;

        if(logger){
            logger->logging(
                "[Query] golden=" + golden_file
                + ", top_k=" + std::to_string(top_k)
                + ", efSearch=" + std::to_string(efSearch) + ", M_exp_search=" + std::to_string(param->M_exp_search),
                true
            );
        }
        else
            printf("[Query] golden=%s, top_k=%d, efSearch=%d, M_exp_search=%d\n", golden_file.c_str(), top_k, efSearch, param->M_exp_search);

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

        std::vector<faiss::idx_t> labels(top_k * 1 * nr_query);
        std::vector<float> dis2(top_k * 1 * nr_query);
        faiss::SearchParametersPathSeer inner_param;
        faiss::pathseer_stats.reset();

        selector->set_filter_batch(query_metadata);

        inner_param.sel = selector;
        inner_param.efSearch = efSearch;
        inner_param.check_relative_distance = true;
        inner_param.M_exp_search = param->M_exp_search;
        inner_param.profile_virtual_overhead = param->profile_virtual_overhead;
        inner_param.print_virutal_overhead = param->print_virtual_overhead;
        inner_param.virtual_overhead = param->virtual_overhead;
        inner_param.coef_a = param->coef_a;
        inner_param.coef_b = param->coef_b;
        if(param->use_mff_only)
            inner_param.only_perform_mff = param->use_mff_only;
        else if(param->use_two_hop_search)
            inner_param.two_hop_search = param->use_two_hop_search;
        else if(param->use_dfn_only_search)
            inner_param.only_perform_dfn = param->use_dfn_only_search;
        

        if(param->print_virtual_overhead)
            printf("[efSearch] %d\n", efSearch);

        int64_t start_time = timing.get_us_time();
        index->search(nr_query, query_data, top_k, dis2.data(), labels.data(), &inner_param);
        int64_t end_time = timing.get_us_time();

        double avg_recall = recall.get_recall_batch(nr_query, labels.data(), top_k);

        char output_content[1024];
        sprintf(output_content, "[Result] Nr_query=%u, total_time=%ld us, Tput=%.2f ops, avg_filter_first=%lu, avg_filter=%lu, avg_dist_first=%lu, avg_dist=%lu, avg_step_first=%lu, avg_step=%lu, avg_recall=%.2f %%, dist_prof_ns=%lld, filter_prof_ns=%lld\n", nr_query, (end_time-start_time), nr_query * 1e6 / (end_time - start_time), faiss::pathseer_stats.nfilter_first / nr_query, faiss::pathseer_stats.nfilter / nr_query, faiss::pathseer_stats.ndis_first / nr_query, faiss::pathseer_stats.ndis / nr_query, faiss::pathseer_stats.nhop_first / nr_query, faiss::pathseer_stats.nhop / nr_query, avg_recall * 100, index->get_profiled_dist_comp_ns(), index->get_profiled_filter_ns());

        logger->logging(std::string(output_content));

        if(selector)
            delete selector;

        return avg_recall * 100;

    }

    // query at an automatically found efSearch
    float auto_batch_query(void* searchParameter, float target_recall, int min_efSearch=0, int max_efSearch=5000) {

        // this means the search terminate early, quit directly
        if(max_efSearch==-1){
            logger->loggingf("[PointSearch] Nr_query=%u, total_time=%ld us, Tput=%.2f ops, avg_dist=%d, avg_filter=%d, avg_recall=%.2f %%\n", nr_query, 0, 0, 0, 0, 0);
            return 0;
        }

        pathseer_searchParameter* param = (pathseer_searchParameter*) searchParameter;

        std::string &golden_file = param->golden_file;
        int top_k = param->top_k;
        enum query_method using_method = param->method;

        // set ground truth
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
        faiss::SearchParametersPathSeer inner_param;
        faiss::pathseer_stats.reset();

        selector->set_filter_batch(query_metadata);

        inner_param.sel = selector;
        inner_param.check_relative_distance = true;
        inner_param.M_exp_search = param->M_exp_search;
        inner_param.profile_virtual_overhead = param->profile_virtual_overhead;
        inner_param.print_virutal_overhead = param->print_virtual_overhead;
        inner_param.virtual_overhead = param->virtual_overhead;
        inner_param.coef_a = param->coef_a;
        inner_param.coef_b = param->coef_b;
        if(param->use_mff_only)
            inner_param.only_perform_mff = param->use_mff_only;
        else if(param->use_two_hop_search)
            inner_param.two_hop_search = param->use_two_hop_search;
        else if(param->use_dfn_only_search)
            inner_param.only_perform_dfn = param->use_dfn_only_search;

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
            faiss::pathseer_stats.reset();
            int64_t start_time = timing.get_us_time();
            index->search(nr_query, query_data, top_k, dis2.data(), labels.data(), &inner_param);
            int64_t end_time = timing.get_us_time();
            double avg_recall = recall.get_recall_batch(nr_query, labels.data(), top_k) * 100;

            if(avg_recall <= target_recall){
                min_efSearch = efSearch;
                recall_min_pos = avg_recall;
                time_min_pos = end_time - start_time;
                ndist_min = faiss::pathseer_stats.ndis;
                nfilter_min = faiss::pathseer_stats.nfilter;
            }
            else{
                max_efSearch = efSearch;
                recall_max_pos = avg_recall;
                time_max_pos = end_time - start_time;
                ndist_max = faiss::pathseer_stats.ndis;
                nfilter_max = faiss::pathseer_stats.nfilter;
            }

        }

        // we cannot let left equal right
        if(min_efSearch==max_efSearch)
            max_efSearch++;

        // one of them may not be calculated, do it
        if(recall_min_pos==-1){
            inner_param.efSearch = min_efSearch;
            faiss::pathseer_stats.reset();
            int64_t start_time = timing.get_us_time();
            index->search(nr_query, query_data, top_k, dis2.data(), labels.data(), &inner_param);
            int64_t end_time = timing.get_us_time();
            recall_min_pos = recall.get_recall_batch(nr_query, labels.data(), top_k) * 100;
            time_min_pos = end_time - start_time;
            ndist_min = faiss::pathseer_stats.ndis;
            nfilter_min = faiss::pathseer_stats.nfilter;
        }
        if(recall_max_pos==-1){
            inner_param.efSearch = max_efSearch;
            faiss::pathseer_stats.reset();
            int64_t start_time = timing.get_us_time();
            index->search(nr_query, query_data, top_k, dis2.data(), labels.data(), &inner_param);
            int64_t end_time = timing.get_us_time();
            recall_max_pos = recall.get_recall_batch(nr_query, labels.data(), top_k) * 100;
            time_max_pos = end_time - start_time;
            ndist_max = faiss::pathseer_stats.ndis;
            nfilter_max = faiss::pathseer_stats.nfilter;
        }

        assert(recall_min_pos<=target_recall && recall_max_pos>=target_recall);

        // if(!(recall_min_pos<=target_recall && recall_max_pos>=target_recall)){
        //     printf("min=%f, max=%f, target=%f, minEfSearch=%d, maxEfSearch=%d\n", recall_min_pos, recall_max_pos, target_recall, min_efSearch, max_efSearch);
        //     assert(false);
        // }

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

    int get_profiled_dist_comp_ns(){
        return index->get_profiled_dist_comp_ns();
    }

    int get_profiled_filter_ns(){
        return index->get_profiled_filter_ns();
    }

    void set_profiled_dist_comp_ns(int ns){
        index->set_profiled_dist_comp_ns(ns);
    }

    void set_profiled_filter_ns(int ns){
        index->set_profiled_filter_ns(ns);
    }

    // Debug helper: for a specific layer (default layer 0), collect distances for each layer.
    void test_entry_distance(void* searchParameter, int layer) {

        pathseer_searchParameter* param = (pathseer_searchParameter*) searchParameter;
        std::string &golden_file = param->golden_file;
        recall.set_metadata(golden_file);

        char output[1024];
        sprintf(output, "test_entry_distance of layer %d", layer);

        if(logger){
            logger->logging(std::string(output));
        }
        else
            printf("%s\n", output);

        // Start search.
        float* query = query_data;
        for(int i=0; i<nr_query; i++){

            faiss::IDSelectorBitmap *selector = new faiss::IDSelectorBitmap(nr_bytes, (uint8_t*)(query_bit_vector + i * nr_bytes));

            // Get the search entry point.
            int64_t closet_golden = recall.get_golden_id(i, 0);
            int max_hop = -1;

            int distance = index->print_entry_distance(query, selector, closet_golden, layer, max_hop);

            query += dimension;

            sprintf(output, "Query_idx=%d, distance=%d, max_hop=%d", i, distance, max_hop);

            if(logger){
                logger->logging(std::string(output));
            }
            else
                printf("%s\n", output);

            delete selector;

        }

        return;

    }

    // PathSeer automatically profiling a suitable virtual overhead
    void profile_virtual_overhead_bound(enum query_method using_method, int sleep_val, std::vector<int>& M_exp_search_params, float target_recall=90, int top_k=10, int time_limit_second=300){

        int64_t all_func_start_time = timing.get_s_time();
        int64_t accum_golden_time = 0;
        int64_t accum_eval_time = 0;

        // step 1: randomly genearte artificial query vectors from the dataset
        int nr_artificial_vector = std::min<int>(1000, nr_vector);
        float* artificial_vector_array = new float[dimension * nr_artificial_vector];

        logger->loggingf("[Nr artifact] %d\n", nr_artificial_vector);

        std::random_device rd; 
        std::mt19937 g(rd());
        std::uniform_int_distribution<int> udist(0, nr_vector-1);

        int64_t generate_query_start_time = timing.get_s_time();
        // we randomly generate test vector by average vector pairs of the dataset
        for(size_t i=0; i<nr_artificial_vector; i++){
            int num1 = udist(g);
            int num2 = udist(g);
            while(num2==num1)
                num2 = udist(g);
            int num3 = udist(g);
            while(num3==num1 || num3==num2)
                num3 = udist(g);

            for(size_t j=0; j<dimension; j++){
                artificial_vector_array[(size_t)(i*dimension + j)] = (vector_data[(size_t)(num1*dimension + j)] + vector_data[(size_t)(num2*dimension + j)] + vector_data[(size_t)(num3*dimension + j)]) / 3;
            }
        }


        // normalize the vector
        normalize(artificial_vector_array, dimension, nr_artificial_vector);
        int64_t generate_query_end_time = timing.get_s_time();

        // int nr_vector_remaining = nr_vector - nr_artificial_vector;

        // memcpy(artificial_vector_array, vector_data + dimension * nr_vector_remaining, sizeof(float) * dimension * nr_artificial_vector);

        // step 2: estimate the avg filtering costs and estimate a suitable sleeping parameter

        // step 2.1: profile the actual filtering costs
        // profiling the filtering costs, which must be done for all methods
        this->profile_ops_cost(using_method, sleep_val);
        int workload_dist_comp_ns = index->get_profiled_dist_comp_ns();
        int workload_filter_time_ns = index->get_profiled_filter_ns();

        logger->loggingf("[Workload profile] dist-comp-ns=%d filter-time-ns=%d\n", workload_dist_comp_ns, workload_filter_time_ns);
        logger->loggingf("[Dist-comp-profile] %d\n", workload_dist_comp_ns);
        logger->loggingf("[Filter-time-profile] %d\n", workload_filter_time_ns);

        int artifact_sleep_val = sleep_val;
        std::string val_method = "DIRECT";
        // for a general case, we should profile a suitable sleep int array value
        // if(using_method!=QUERY_METHOD_INT_ARRAY_SLEEP || using_method!=QUERY_METHOD_INT_ARRAY){
        if(1){

            val_method = "PROFILE";

            // step 2.2: we use a binary search to find a suitable parameter
            // as sleep_val = 0 may have large filtering costs for IntArraySleep, we intialize two selectors
            faiss::IDSelector* selector = nullptr;  // the one we use
            faiss::IDSelector* sel_int_arr = new IDSelectorIntArray(nr_vector);
            faiss:IDSelectorIntArraySleep* sel_int_arr_sleep = new IDSelectorIntArraySleep(nr_vector);
            
            std::mt19937 g2(rd());
            std::uniform_int_distribution<int> dist(1, 100);

            // randomly generate the query metadata
            std::vector<int> artificial_query_metadata(nr_artificial_vector);
            for(int i=0; i<nr_artificial_vector; i++){
                artificial_query_metadata[i] = dist(g2);
            }

            int lower_bound_val = 0;
            int higher_bound_val = 500;

            // firstly, find a suitable higher bound
            selector = sel_int_arr_sleep;
            while(1){

                selector->set_sleep_ns(higher_bound_val);
                selector->set_filter("0");

                auto start = std::chrono::high_resolution_clock::now();
                int final_result = 0;
                for(int i=0; i<nr_artificial_vector; i++){
                    final_result += selector->is_member(artificial_query_metadata[i]);
                }
                auto end = std::chrono::high_resolution_clock::now();
                long long int avg_lat_ns = (long long int)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / nr_artificial_vector;

                if(avg_lat_ns >= workload_filter_time_ns)
                    break;
                else
                    higher_bound_val *= 2;

            }

            // printf("[Info] chosing higher bound %d\n", higher_bound_val);

            // secondly, using binary search to get a suitable sleep val
            int mid = 0;
            while(lower_bound_val < higher_bound_val){

                mid = (lower_bound_val + higher_bound_val) / 2;

                // printf("[BSearch] [%d, %d], mid=%d\n", lower_bound_val, higher_bound_val, mid);

                if(!mid)
                    selector = sel_int_arr;
                else{
                    selector = sel_int_arr_sleep;
                    selector->set_sleep_ns(mid);
                }

                selector->set_filter("0");

                // we tests five times and use the middle one to tackle the latency spike
                std::priority_queue<long long int, std::vector<long long int>, std::greater<long long int>> pq;

                int try_times = 5;
                for(int t=0; t<try_times; t++){
                    auto start = std::chrono::high_resolution_clock::now();
                    int final_result = 0;
                    for(int i=0; i<nr_artificial_vector; i++){
                        final_result += selector->is_member(artificial_query_metadata[i]);
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    long long int avg_lat_ns = (long long int)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / nr_artificial_vector;

                    pq.push(avg_lat_ns);

                    // printf("[One profile] lat is %lld\n", avg_lat_ns);

                }

                try_times /= 2;
                while(try_times-- > 0){
                    pq.pop();
                }

                long long int avg_lat_ns = pq.top();

                // if(avg_lat_ns == workload_filter_time_ns)
                // as the measurement has uncertainty, we do not finish when they are strict equal
                if(std::abs(avg_lat_ns - workload_filter_time_ns) <= workload_filter_time_ns * 0.05){
                    // printf("[Break] gonna break, now lat=%lld, target=%d\n", avg_lat_ns, workload_filter_time_ns);
                    break;
                }
                else if(avg_lat_ns < workload_filter_time_ns)
                    lower_bound_val = mid + 1;
                else
                    higher_bound_val = mid;
            }

            artifact_sleep_val = mid;

            // printf("[Res] chosing sleep val %d, low=%d, high=%d, mid=%d\n", artifact_sleep_val, lower_bound_val, higher_bound_val, mid);

            delete selector;

        }

        // artifact_sleep_val = 20;

        logger->loggingf("[Sleep_val] chosing sleep val %d by %s\n", artifact_sleep_val, val_method.c_str());

        // step 3: randomly generate integer selector for each of the Sp, and derive the golden value
        std::vector<int> chosen_max_val = {2, 4, 8, 16, 32, 64};
        std::vector<int> suitable_virtual_overhead_all;

        // record every pairs of profiled result
        // std::vector<std::vector<virtual_overhead_pair>> prof_result_all;
        // record the shortest time for each exp
        std::vector<float> shortest_eval_time_inv;

        for(int k=0; k<chosen_max_val.size(); k++){

            int max_val = chosen_max_val[k];
            logger->loggingf("[Max_val] %d\n", max_val);

            // 3.1 generate random dataset metadata and query metadata
            std::vector<int> dataset_metadata(nr_vector);
            std::vector<std::string> _query_metadata(nr_artificial_vector);
            std::mt19937 g3(rd());
            std::uniform_int_distribution<int> dist(1, max_val);
            // dataset vectors
            for(int i=0; i<nr_vector; i++){
                dataset_metadata[i] = dist(g3);
            }
            // query vectors
            for(int i=0; i<nr_artificial_vector; i++){
                _query_metadata[i] = std::to_string(dist(g3));
            }

            // 3.2 generate the golden
            int64_t golden_start_time = timing.get_s_time();
            faiss::IndexFlat golden_index(dimension, faiss::METRIC_INNER_PRODUCT);

            // find the golden in a parallel way
            std::vector<float> dist_arr(top_k * nr_artificial_vector, -1);
            std::vector<faiss::idx_t> label_arr(top_k * nr_artificial_vector, -1);

            // for the golden generation, we don not need to set the sleep time as it does not influece the result
            faiss::IDSelector* sel = new IDSelectorIntArray(dataset_metadata);

            faiss::SearchParameters param;
            param.sel = sel;

            // parallel search the golden
            sel->set_filter_batch(_query_metadata);

            #pragma omp parallel for
            for(int i=0; i<nr_artificial_vector; i++){

                TopKHeapArray heap(top_k);
                float* the_query = artificial_vector_array + dimension * i;
                // for each dataset vector
                for(size_t iii=0; iii<nr_vector; iii++){
                    if(sel->is_member(iii, i)){
                        float _dist = fvec_inner_product_avx2(the_query, vector_data + (size_t) dimension * iii, dimension);
                        heap.add(_dist, iii);
                    }
                }
                // write-out the golden
                int pos_end = heap.size() - 1;
                while(!heap.empty()){
                    Pair top = heap.pop();
                    dist_arr[pos_end + top_k*i] = top.score;
                    label_arr[pos_end + top_k*i] = top.id;
                    pos_end--;
                }
            }

            delete sel;

            // check the golden
            for(int i=0; i<nr_artificial_vector*top_k; i++){
                if(label_arr[i]==-1){
                    printf("Error: find empty golden!\n");
                    assert(false);
                }
            }


            int64_t golden_end_time = timing.get_s_time();
            accum_golden_time += (golden_end_time - golden_start_time);

            // 3.3 search for each M_exp individually
            if(artifact_sleep_val){
                sel = new IDSelectorIntArraySleep(dataset_metadata);
                // sel->set_sleep_ns(artifact_sleep_val); // we  initially does not use this
                sel->set_sleep_ns(0);
            }
            else
                sel = new IDSelectorIntArray(dataset_metadata);
            std::vector<faiss::idx_t> retrieved_label_arr(top_k * nr_artificial_vector);
            // set the filter for batch query
            sel->set_filter_batch(_query_metadata);
            // set the golden from above
            recall.set_metadata(label_arr, top_k);
            // search params
            faiss::SearchParametersPathSeer inner_param;
            inner_param.sel = sel;
            inner_param.check_relative_distance = true;
            inner_param.profile_virtual_overhead = false;
            inner_param.M_exp_search = 0;

            // record the virtual overhead correlated to the minimal eval time
            int suitable_virtual_overhead = 0;
            int64_t minimal_test_time = INT64_MAX;

            int64_t eval_start_time = timing.get_s_time();
            // for each virtual overhead boundary, profiling
            int min_M_exp = M_exp_search_params[0];
            int max_M_exp = M_exp_search_params[M_exp_search_params.size()-1];

            int min_VO = min_M_exp * workload_filter_time_ns;
            int max_VO = max_M_exp * (workload_filter_time_ns + workload_dist_comp_ns * 0.5);
            int VO_delta = (max_VO - min_VO) / 10;

            for(int vo = min_VO; vo<=max_VO; vo += VO_delta){

                int VO_sum = 0;
                int64_t test_time_sum = 0;
                int nr_test_times = 3;

                logger->loggingf("[VO] %d\n", vo);

                // prof_result_all.push_back(std::vector<virtual_overhead_pair> ());

                inner_param.virtual_overhead = vo;

                // 3.4 search multiple times to get the efSearch when recall is the target.
                // we will evaluate the time of each test. If it exceeds the time threshold, we mark this configuration as TLE and pass it
                int efSearch = top_k;
                inner_param.efSearch = efSearch;

                float candidate_recall = 0;
                size_t this_test_VO = 0;

                int suitable_efSearch = efSearch;
                float prev_recall = 0;
                // find the lower bound and upper bound of the recall
                int efSearch_lowerbound = -1;
                int efSearch_upperbound = -1;

                // temporarily set the sleep ns to 0
                if(artifact_sleep_val)
                    sel->set_sleep_ns(0);

                while(1){

                    inner_param.efSearch = efSearch;

                    faiss::pathseer_stats.reset();
                    int64_t start_time = timing.get_s_time();
                    index->search(nr_artificial_vector, artificial_vector_array, top_k, dist_arr.data(), retrieved_label_arr.data(), &inner_param);
                    int64_t end_time = timing.get_s_time();

                    double this_recall = recall.get_recall_batch(nr_artificial_vector, retrieved_label_arr.data(), top_k) * 100;
                    
                    if(this_recall - prev_recall < 0.01){
                        printf("Error: search stacked without increasing recall rates, from %f to %f.\n", prev_recall, this_recall);
                        assert(false);
                    }

                    if(this_recall <= target_recall){
                        efSearch_lowerbound = efSearch;
                    }

                    if(this_recall > target_recall){
                        efSearch_upperbound = efSearch;
                        candidate_recall = this_recall;
                        break;
                    }

                    // printf("efSearch=%d, r=%f, pr=%f, time=%ld s\n", efSearch, this_recall, prev_recall, end_time-start_time);

                    // if the recall does not satisfy target, but the time exceeds, we early stop
                    if(end_time-start_time >= time_limit_second){
                        logger->loggingf("[EarlyStop] TLE threshold %d s, early stop\n", time_limit_second);
                        test_time_sum = int(time_limit_second * 1e6);
                        VO_sum = vo;
                        goto early_stop;
                    }


                    // set the next efSearch according to the recall delta
                    // only from the second on
                    // an optimized algorithm to find an approaching target recall: the recall is nearly linear to the log of efSearch.
                    // int efMultiplier = 2;
                    // if(prev_recall==0){
                    //     efMultiplier = 2;
                    //     init_recall = this_recall;
                    // }
                    // else{
                    //     int efMul = std::max<int>((int) (target_recall - init_recall) / (this_recall - init_recall), 1);
                    //     printf("[efMul] %d\n", efMul);
                    //     efMultiplier = std::max<int>(2, std::pow(2, efMul));
                    // }

                    // printf("efMultiplier=%d, efSearch=%d, r=%f, pr=%f, ir=%f, time=%ld s\n", efMultiplier, efSearch, this_recall, prev_recall, init_recall, end_time-start_time);

                    // we gradually increase the efSearch anyway
                    efSearch = (int) (efSearch * 2);
                    prev_recall = this_recall;

                }

                if(efSearch_upperbound==-1){
                    printf("[Error] cannot find profiling efSearch of target recall!\n");
                    assert(false);
                }

                // the minimal efSearch has recall higher than target, just use it
                if(efSearch_lowerbound==-1){
                    assert(efSearch_upperbound == top_k);
                    suitable_efSearch = top_k;
                }
                else{
                    // search an accurate efSearch of the target recall
                    float recall_min_pos = -1;
                    float recall_max_pos = -1;

                    while(efSearch_lowerbound + 1 < efSearch_upperbound){

                        int efSearch = (efSearch_lowerbound + efSearch_upperbound) / 2;

                        inner_param.efSearch = efSearch;
                        index->search(nr_artificial_vector, artificial_vector_array, top_k, dist_arr.data(), retrieved_label_arr.data(), &inner_param);
                        double avg_recall = recall.get_recall_batch(nr_artificial_vector, retrieved_label_arr.data(), top_k) * 100;

                        if(avg_recall <= target_recall){
                            efSearch_lowerbound = efSearch;
                            recall_min_pos = avg_recall;
                        }
                        else{
                            efSearch_upperbound = efSearch;
                            recall_max_pos = avg_recall;
                        }

                    }

                    // we cannot let left equal right
                    if(efSearch_lowerbound==efSearch_upperbound)
                        efSearch_upperbound++;

                    // one of them may not be calculated, do it
                    if(recall_min_pos==-1){
                        inner_param.efSearch = efSearch_lowerbound;
                        index->search(nr_artificial_vector, artificial_vector_array, top_k, dist_arr.data(), retrieved_label_arr.data(), &inner_param);
                        recall_min_pos = recall.get_recall_batch(nr_artificial_vector, retrieved_label_arr.data(), top_k) * 100;
                    }
                    if(recall_max_pos==-1){
                        inner_param.efSearch = efSearch_upperbound;
                        index->search(nr_artificial_vector, artificial_vector_array, top_k, dist_arr.data(), retrieved_label_arr.data(), &inner_param);
                        recall_max_pos = recall.get_recall_batch(nr_artificial_vector, retrieved_label_arr.data(), top_k) * 100;
                    }

                    // assert(recall_min_pos<=target_recall && recall_max_pos>=target_recall);
                    if (!(recall_min_pos<=target_recall && recall_max_pos>=target_recall)){
                        printf("Recall min %f, recall max %f, target %f\n", recall_min_pos, recall_max_pos, target_recall);
                        assert(false);
                    }

                    if(target_recall - recall_min_pos <= recall_max_pos - target_recall){
                        suitable_efSearch = efSearch_lowerbound;
                        candidate_recall = recall_min_pos;
                    }
                    else{
                        suitable_efSearch = efSearch_upperbound;
                        candidate_recall = recall_max_pos;
                    }
                }

                logger->loggingf("[Recall] chosen recall is %f, efSearch is %d\n", candidate_recall, suitable_efSearch);

                // 3.5 test multiple times with the efSearch to get a stable performance, as well as recording each virtual overhead

                inner_param.efSearch = suitable_efSearch;

                // resort the actual sleep ns
                if(artifact_sleep_val)
                    sel->set_sleep_ns(artifact_sleep_val);

                for(int i=0; i<nr_test_times; i++){

                    faiss::pathseer_stats.reset();

                    int64_t start_time = timing.get_us_time();
                    index->search(nr_artificial_vector, artificial_vector_array, top_k, dist_arr.data(), retrieved_label_arr.data(), &inner_param);
                    int64_t end_time = timing.get_us_time();

                    size_t this_test_VO = vo;

                    logger->loggingf("[Measure] test %d, time_us %lld, virtual_overhead %lu\n", i, end_time - start_time, this_test_VO);

                    VO_sum += this_test_VO;
                    test_time_sum += end_time - start_time;

                }

                VO_sum /= nr_test_times;
                test_time_sum /= nr_test_times;

             early_stop:

                logger->loggingf("[Virtual_overhead_each_summary] avg_time=%d virtual_overhead=%d\n", test_time_sum, VO_sum);

                if(test_time_sum < minimal_test_time){
                    minimal_test_time = test_time_sum;
                    suitable_virtual_overhead = VO_sum;
                }

                // prof_result_all[j].push_back(virtual_overhead_pair(nr_artificial_vector * 1e6 / test_time_sum, VO_sum));
                // printf("[PushSum] %f\n", 1.0 / test_time_sum);

            }
            int64_t eval_end_time = timing.get_s_time();
            accum_eval_time += (eval_end_time - eval_start_time);

            // step 4: for each Sp, query multiple times to derive the suitable virtual overhead
            suitable_virtual_overhead_all.push_back(suitable_virtual_overhead);
            shortest_eval_time_inv.push_back(nr_artificial_vector * 1e6 / minimal_test_time);

            // printf("[TT] push to %d of %f\n", k, shortest_eval_time_inv[k]);

        }

        // step 5: average all suitable virtual overhead and output
        std::string output_str = "[Candidate_virtual_overhead]";
        for(auto it : suitable_virtual_overhead_all){
            output_str = output_str + " " + std::to_string(it);
        }

        logger->logging(output_str, true);

        int final_virtual_overhead = static_cast<double>(std::accumulate(suitable_virtual_overhead_all.begin(), suitable_virtual_overhead_all.end(), 0)) / suitable_virtual_overhead_all.size();

        // output the avg virtual overhead
        logger->loggingf("[Avg_virtual_overhead] %d\n", final_virtual_overhead);

        delete artificial_vector_array;

        int64_t all_func_end_time = timing.get_s_time();

        logger->loggingf("[GenerateQueryTime] %lld s\n", generate_query_end_time-generate_query_start_time);
        logger->loggingf("[GoldenTime] %lld s\n", accum_golden_time);
        logger->loggingf("[EvalTime] %lld s\n", accum_eval_time);
        logger->loggingf("[WholeTime] %lld s\n", all_func_end_time - all_func_start_time);

    }

    // this function is used to load best virtual overhead from the file
    // mode=0: loading the low-perf-loss overhead
    // mode=1: loading the simple avg overhead
    void load_virtual_overhead(std::string filepath, int& dist_comp_ns, int& filter_time_ns, int& virtual_overhead, float& coef_a, float& coef_b){

        std::ifstream file(filepath);
        if (!file.is_open()) {
            printf("[Info] Virtual overhead cannot be loaded, gonna using fixed virtual overhead.\n");
            virtual_overhead = 0;
            coef_a = 0;
            coef_b = 0;
            return;
        }

        dist_comp_ns = -1;
        filter_time_ns = -1;

        std::string line;

        while (std::getline(file, line)) {

            // the virtual overhead fitted by equality assumption
            if (line.find("[Best_virtual_overhead]") == 0) {

                std::istringstream iss(line);
                std::string prefix;
                int value;
                
                iss >> prefix >> value;
                
                if (iss.fail()) {
                    throw std::runtime_error("Cannot get overhead from:" + line);
                }
                
                virtual_overhead = value;

            }
            // the virtual overhead fitted by linear assumption
            else if (line.find("[LoadCoefficient]") == 0) {

                std::istringstream iss(line);
                std::string prefix;
                float a, b;
                
                iss >> prefix >> a >> b;
                
                if (iss.fail()) {
                    throw std::runtime_error("Cannot get coefficient from:" + line);
                }
                
                coef_a = a;
                coef_b = b;

            }
            // get the filter time ns and dist comp time ns
            else if(line.find("[Dist-comp-profile]") == 0) {

                std::istringstream iss(line);
                std::string prefix;
                int value;
                
                iss >> prefix >> value;
                
                if (iss.fail()) {
                    throw std::runtime_error("Cannot get dist-comp-ns from:" + line);
                }
                
                dist_comp_ns = value;

            }
            else if(line.find("[Filter-time-profile]") == 0) {

                std::istringstream iss(line);
                std::string prefix;
                int value;
                
                iss >> prefix >> value;
                
                if (iss.fail()) {
                    throw std::runtime_error("Cannot get filter-time-ns from:" + line);
                }
                
                filter_time_ns = value;

            }

        }

        if(dist_comp_ns==-1 || filter_time_ns==-1){
            printf("[Error] Please check the dist-comp and filter time in file!\n");
            assert(false);
        }

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

        if(index)
            delete index;

        index = dynamic_cast<faiss::IndexPathSeerFlat*>(faiss::read_index(persist_path));

    }

    int get_nr_query(){
        return this->nr_query;
    }

};

#endif