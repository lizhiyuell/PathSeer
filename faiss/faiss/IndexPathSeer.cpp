/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexPathSeer.h>

#include <omp.h>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <queue>
#include <unordered_set>

#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

// # added
#include <sys/time.h>
#include <stdio.h>
#include <iostream>

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {

using MinimaxHeap = PathSeer::MinimaxHeap;
using storage_idx_t = PathSeer::storage_idx_t;
using NodeDistFarther = PathSeer::NodeDistFarther;

PathSeerStats pathseer_stats;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */

struct NegativeDistanceComputer : DistanceComputer {

    /// owned by this
    DistanceComputer* basedis;

    explicit NegativeDistanceComputer(DistanceComputer* basedis)
            : basedis(basedis) {}

    void set_query(const float* x) override {
        basedis->set_query(x);
    }

    /// compute distance of vector i to current query
    float operator()(idx_t i) override {
        return -(*basedis)(i);
    }

    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override {
        return -basedis->symmetric_dis(i, j);
    }

    void prefetch_vector(idx_t i) override {
        basedis->prefetch_vector(i);
    }

    virtual ~NegativeDistanceComputer() {
        delete basedis;
    }
};

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (storage->metric_type == METRIC_INNER_PRODUCT) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

// TODO
void pathseer_add_vertices(
        IndexPathSeer& index_pathseer,
        size_t n0,
        size_t n,
        const float* x,
        bool verbose,
        bool preset_levels = false) {
    size_t d = index_pathseer.d;
    PathSeer& pathseer = index_pathseer.pathseer;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("pathseer_add_vertices: adding %zd elements on top of %zd "
               "(preset_levels=%d)\n",
               n,
               n0,
               int(preset_levels));
    }

    if (n == 0) {
        return;
    }

    // randomly generate the layer for each node
    int max_level = pathseer.prepare_level_tab(n, preset_levels);

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

    // add vectors from highest to lowest level
    std::vector<int> hist; // this record the number of node in each level
    std::vector<int> order(n);

    { // make buckets with vectors of the same level

        // build histogram
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = pathseer.levels[pt_id] - 1;
            while (pt_level >= hist.size())
                hist.push_back(0);
            hist[pt_level]++;
        }

        // accumulate
        // offsets record the accumulated numbers of vectors in each level
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        // this put the pt of small levels to front, and put the pt of large levels to end
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = pathseer.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }

    }

    idx_t check_period = InterruptCallback::get_period_hint(
            max_level * index_pathseer.d * pathseer.efConstruction);

    { // perform add
        RandomGenerator rng2(789);

        int i1 = n;

        for (int pt_level = hist.size() - 1; pt_level >= 0; pt_level--) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
            }

            // random permutation to get rid of dataset order bias
            for (int j = i0; j < i1; j++)
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

            bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(ntotal);

                DistanceComputer* dis =
                        storage_distance_computer(index_pathseer.storage);
                ScopeDeleter1<DistanceComputer> del(dis);
                int prev_display =
                        verbose && omp_get_thread_num() == 0 ? 0 : -1;
                size_t counter = 0;

                // here we should do schedule(dynamic) but this segfaults for
                // some versions of LLVM. The performance impact should not be
                // too large when (i1 - i0) / num_threads >> 1
#pragma omp for schedule(static)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query(x + (pt_id - n0) * d);

                    // cannot break
                    if (interrupt) {
                        continue;
                    }

                    pathseer.add_with_locks(*dis, pt_level, pt_id, locks, vt);

                    if (prev_display >= 0 && i - i0 > prev_display + 100) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }
                    if (counter % check_period == 0) {
                        if (InterruptCallback::is_interrupted()) {
                            interrupt = true;
                        }
                    }
                    counter++;
                }
            }
            if (interrupt) {
                FAISS_THROW_MSG("computation interrupted");
            }
            i1 = i0;
        }
        FAISS_ASSERT(i1 == 0);
    }
    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }

    for (int i = 0; i < ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }
}


} // namespace

/**************************************************************
 * IndexPathSeer implementation
 **************************************************************/

IndexPathSeer::IndexPathSeer(int d, int M, int M_exp, MetricType metric)
        : Index(d, metric),
          pathseer(M, M_exp),
          own_fields(false),
          storage(nullptr)
          /* reconstruct_from_neighbors(nullptr)*/ {}

IndexPathSeer::IndexPathSeer(Index* storage, int M, int M_exp)
        : Index(storage->d, storage->metric_type),
          pathseer(M, M_exp), // TOOD pathseer needs to keep metadata now
          own_fields(false),
          storage(storage)
          /* reconstruct_from_neighbors(nullptr) */ {}

IndexPathSeer::~IndexPathSeer() {
    if (own_fields) {
        delete storage;
    }
}

void IndexPathSeer::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexPathSeerFlat (or variants) instead of IndexPathSeer directly");
    // pathseer structure does not require training
    storage->train(n, x);
    is_trained = true;
}


void IndexPathSeer::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexPathSeerFlat (or variants) instead of IndexPathSeer directly");
    const SearchParametersPathSeer* params = nullptr;

    int efSearch = pathseer.efSearch;
    if (params_in) {
        params = dynamic_cast<const SearchParametersPathSeer*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
        efSearch = params->efSearch;
    }
    size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, ndis_first = 0, nhop = 0, nhop_first = 0, nfilter = 0, nfilter_first = 0, virtual_overhead = 0;
    
    idx_t check_period =
            InterruptCallback::get_period_hint(pathseer.max_level * d * efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);

            DistanceComputer* dis = storage_distance_computer(storage);
            ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for reduction(+ : n1, n2, n3, ndis, ndis_first, nhop, nhop_first, nfilter, nfilter_first, virtual_overhead)
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                maxheap_heapify(k, simi, idxi);

                PathSeerStats stats = pathseer.search(*dis, k, idxi, simi, vt, i, params);

                n1 += stats.n1;
                n2 += stats.n2;
                n3 += stats.n3;
                ndis += stats.ndis;
                ndis_first += stats.ndis_first;
                nhop += stats.nhop;
                nhop_first += stats.nhop_first;

                nfilter += stats.nfilter;
                nfilter_first += stats.nfilter_first;

                virtual_overhead += stats.virtual_overhead;

                maxheap_reorder(k, simi, idxi);

            }
        }
        InterruptCallback::check();
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }

    // calculate the average virtual overhead per query
    virtual_overhead /= n;

    pathseer_stats.combine({n1, n2, n3, ndis, ndis_first, nfilter, nfilter_first, nhop, nhop_first, virtual_overhead});

}

void IndexPathSeer::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexPathSeerFlat (or variants) instead of IndexPathSeer directly");
    FAISS_THROW_IF_NOT(is_trained);
    int n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    pathseer_add_vertices(*this, n0, n, x, verbose, pathseer.levels.size() == ntotal);
}

void IndexPathSeer::reset() {
    pathseer.reset();
    storage->reset();
    ntotal = 0;
}

void IndexPathSeer::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}

// profile the dist-comp cost
void IndexPathSeer::profile_dist_comp_cost(){

    // we only profile when needed
    if(pathseer.profilier->dist_comp_cost_ns)
        return;

    int nr_vector = pathseer.levels.size();

    // generate random indices
    std::random_device rd;
    std::mt19937 g(rd());
    int nr_profiled = nr_vector * 1e-2;
    
    int *indices = new int[nr_vector];

    // we leave the first vector alone, as it is used to compute distance with others
    for(int i=0; i<nr_vector; i++){
        indices[i] = i+1;
    }

    std::shuffle(indices, indices + nr_vector, g);

    DistanceComputer* dis = storage_distance_computer(storage);

    // profiling the distance computation time
    long long int total_time = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<nr_profiled; i++){
        volatile float _dist = dis->symmetric_dis(0, indices[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();

    long long int avg_lat_ns = (long long int)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / nr_profiled;

    pathseer.profilier->dist_comp_cost_ns = avg_lat_ns;
    printf("[Profile] profile dist-comp-cost with %d vectors, %lld ns\n", nr_profiled, avg_lat_ns);

    delete indices;

}

// profile the filter cost
void IndexPathSeer::profile_filter_cost(IDSelector* sel){

    // we only profile when needed
    if(pathseer.profilier->filter_cost_ns)
        return;

    int nr_vector = pathseer.levels.size();

    // generate random indices
    std::random_device rd;
    std::mt19937 g(rd());
    int nr_profiled = nr_vector * 1e-2;
    
    int *indices = new int[nr_vector];

    for(int i=0; i<nr_vector; i++){
        indices[i] = i;
    }

    std::shuffle(indices, indices + nr_vector, g);

    // profiling the distance computation time
    long long int total_time = 0;
    int final_result = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<nr_profiled; i++){
        final_result += sel->is_member(indices[i], 0);
    }
    auto end = std::chrono::high_resolution_clock::now();

    long long int avg_lat_ns = (long long int)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / nr_profiled;

    delete indices;

    pathseer.profilier->filter_cost_ns = avg_lat_ns;
    printf("[Profile] profile metadata-filter-cost with %d vectors, %lld ns\n", nr_profiled, avg_lat_ns);

}

long long int IndexPathSeer::get_profiled_dist_comp_ns(){
    return pathseer.profilier->dist_comp_cost_ns;
}

long long int IndexPathSeer::get_profiled_filter_ns(){
    return pathseer.profilier->filter_cost_ns;
}

void IndexPathSeer::set_profiled_dist_comp_ns(long long int ns){
    pathseer.profilier->dist_comp_cost_ns = ns;
}

void IndexPathSeer::set_profiled_filter_ns(long long int ns){
    pathseer.profilier->filter_cost_ns = ns;
}

/**************************************************************
 * IndexPathSeerFlat implementation
 **************************************************************/

IndexPathSeerFlat::IndexPathSeerFlat(int d, int M, int M_exp, MetricType metric)
        : IndexPathSeer(new IndexFlat(d, metric), M, M_exp) {
    own_fields = true;
    is_trained = true;
}

} // namespace faiss
