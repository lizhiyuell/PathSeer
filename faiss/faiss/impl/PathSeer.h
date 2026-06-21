/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <queue>
#include <unordered_set>
#include <vector>
#include <set>
#include <unordered_map>

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>

#include <stdio.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/prefetch.h>
#include <faiss/impl/platform_macros.h>

#include <x86intrin.h>
#include <chrono>

#include <random>

#include <numeric>
#include <cmath>

#include <cstring>
#include <limits>

// the ways for index construction
// #define PATHSEER_CONSTRUCTION_METHOD 0  // basic slow construction method
// #define PATHSEER_CONSTRUCTION_METHOD 1 // don't put the dist vectors into the expanding zone, which perform worse
// #define PATHSEER_CONSTRUCTION_METHOD 2 // + batch adding of source points
// #define PATHSEER_CONSTRUCTION_METHOD 3 // + the optimized construction method without altering the accuracy

// whether using the prefetched selector
#define ENABLE_PREFETCH


namespace faiss {

struct VisitedTable;
struct DistanceComputer; // from AuxIndexStructures
struct PathSeerStats;

struct SearchParametersPathSeer : SearchParameters {
    int efSearch = 16;
    bool check_relative_distance = true;
    int M_exp_search = 16;
    bool profile_virtual_overhead = false; // whether profiling the virtual overhead
    bool print_virutal_overhead = false; // whether printf the virtual overhead
    int virtual_overhead = 0; // we may set the virtual overhead by ourselves
    float coef_a = 0;
    float coef_b = 0;
    bool only_perform_mff = false; // if this is set true, the search only uses mff instead of dcf search
    bool two_hop_search = false; // if this is set true, the search is mff-only two-hop search. this parameter has lower priority than the "only_perform_mff". Note that due to benchmark purpose, we will print the dist of each filtered vectors to stdout
    bool only_perform_dfn = false; // if this is set true, the search is dfn-only search

    ~SearchParametersPathSeer() {}
};

struct static_profile_params {
    long long int dist_comp_cost_ns = 0;
    long long int filter_cost_ns = 0;
};

struct PathSeer {
    /// internal storage of vectors (32 bits: this is expensive)
    using storage_idx_t = int32_t;

    typedef std::pair<float, storage_idx_t> Node; // for heaps with distance, storage

    // stores storage_id of a node
    typedef storage_idx_t NeighNode;

    /* Heap structure that allows fast */
    struct MinimaxHeap {
        int n;
        int k;
        int nvalid;

        std::vector<storage_idx_t> ids;
        std::vector<float> dis;
        typedef faiss::CMax<float, storage_idx_t> HC;

        explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0), ids(n), dis(n) {}

        // copy-based construction function
        MinimaxHeap(const MinimaxHeap& other)
        : n(other.n), k(other.k), nvalid(other.nvalid), ids(other.ids), dis(other.dis) {}

        int push(storage_idx_t i, float v);

        float max() const;

        int size() const;

        void clear();

        int pop_min(float* vmin_out = nullptr);

        int count_below(float thresh);

    };

    /// to sort pairs of (id, distance) from nearest to fathest or the reverse
    struct NodeDistCloser {
        float d;
        int id;
        NodeDistCloser(float d, int id) : d(d), id(id) {}
        NodeDistCloser() : d(0), id(0) {}
        bool operator<(const NodeDistCloser& obj1) const {
            return d < obj1.d;
        }
    };

    struct NodeDistFarther {
        float d;
        int id;
        NodeDistFarther(float d, int id) : d(d), id(id) {}
        NodeDistFarther() : d(0), id(0) {}
        bool operator<(const NodeDistFarther& obj1) const {
            return d > obj1.d;
        }
    };

    /// assignment probability to each layer (sum=1)
    std::vector<double> assign_probas;

    /// number of neighbors stored per layer (cumulative), should not
    /// be changed after first add
    std::vector<int> cum_nneighbor_per_level;

    /// level of each vector (base level = 1), size = ntotal
    std::vector<int> levels;

    /// offsets[i] is the offset in the neighbors array where vector i is stored
    /// size ntotal + 1
    std::vector<size_t> offsets;

    /// neighbors[offsets[i]:offsets[i+1]] is the list of neighbors of vector i
    /// for all levels. this is where all storage goes.
    std::vector<NeighNode> neighbors; // changed to add metadata

    // store the boundary of the dist-based neighbors.
    std::vector<int> offset_neighbor;
    std::vector<int> neighbor_boundaries;

    // record whether use the only-expanding mode for comparison
    bool use_only_expanding_idx = false;

    // record whether use the optimized building method. As it is only used for benchmark purpose, we do not persist this value and default it as true

    bool use_optimized_building = true;

    /// entry point in the search structure (one of the points with maximum
    /// level
    storage_idx_t entry_point;

    faiss::RandomGenerator rng;

    int M;

    int M_exp; // M_expanding for construction

    // overhead profilter
    struct static_profile_params* profilier = nullptr;

    /// maximum level
    int max_level;

    /// expansion factor at construction time, for dist-oriented vectors
    int efConstruction = 40;
    // for expanding area
    int efConstructionExp = 160;

    /// expansion factor at search time
    int efSearch;

    /// during search: do we check whether the next best distance is good
    /// enough?
    bool check_relative_distance = true;

    /// number of entry points in levels > 0.
    int upper_beam;

    /// use bounded queue during exploration
    bool search_bounded_queue = true;

    // methods that initialize the tree sizes

    /// initialize the assign_probas and cum_nneighbor_per_level to
    /// have 2*M links on level 0 and M links on levels > 0
    void set_default_probas(int M, float levelMult, int M_exp);

    /// set nb of neighbors for this level (before adding anything)
    void set_nb_neighbors(int level_no, int n);

/// nb of neighbors for this level
    int nb_neighbors(int layer_no) const;

    /// cumumlative nb up to (and excluding) this level
    int cum_nb_neighbors(int layer_no) const;

    /// range of entries in the neighbors table of vertex no at layer_no
    void neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end) const;

    int get_neighbor_dist_boundary(idx_t no, int layer_no) const;
    void set_neighbor_dist_boundary(idx_t no, int layer_no, int new_boundary);

    /// only mandatory parameter: nb of neighbors
    // explicit HNSW(int M = 32);
    explicit PathSeer(int M, int M_exp);

    ~PathSeer();

    /// pick a random level for a new point
    int random_level();

    /// add n random levels to table (for debugging...)
    void fill_with_random_links(size_t n);

    void add_links_starting_from(
            DistanceComputer& ptdis,
            storage_idx_t pt_id,
            storage_idx_t nearest,
            float d_nearest,
            int level,
            omp_lock_t* locks,
            VisitedTable& vt);

    void add_links_starting_from_only_expanding(
            DistanceComputer& ptdis,
            storage_idx_t pt_id,
            storage_idx_t nearest,
            float d_nearest,
            int level,
            omp_lock_t* locks,
            VisitedTable& vt);

    /** add point pt_id on all levels <= pt_level and build the link
     * structure for them. */
    void add_with_locks(
            DistanceComputer& ptdis,
            int pt_level,
            int pt_id,
            std::vector<omp_lock_t>& locks,
            VisitedTable& vt);

    /// search interface for 1 point, single thread
    PathSeerStats search(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            VisitedTable& vt,
            int pos,
            const SearchParametersPathSeer* params = nullptr) const;
    
    /**************************************************************
    * PathSeer HYBRID INDEX
    **************************************************************/

    PathSeerStats adaptive_hybrid_search(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            VisitedTable& vt,
            const SearchParametersPathSeer* params = nullptr) const;

    /**************************************************************
    **************************************************************/
 
    void reset();

    void clear_neighbor_tables(int level);

    int prepare_level_tab(size_t n, bool preset_levels = false);

    void shrink_neighbor_list(
            DistanceComputer& qdis,
            std::priority_queue<NodeDistFarther>& input,
            std::vector<NodeDistFarther>& output,
            int max_size, int level = 0);

    // the original shrinking function of HNSW
    static void HNSW_shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistFarther>& input,
        std::vector<NodeDistFarther>& output,
        int max_size,
        std::vector<NodeDistFarther>& outsiders,
        bool keep_max_size_level0);
    
    // modififed shrinking function of PathSeer
    static void PathSeer_shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistFarther>& input,
        std::priority_queue<NodeDistCloser>& other,
        std::vector<NodeDistFarther>& output,
        int max_size);

    // asynchrnously profiling the dist-comp cost
    void profile_dist_comp_cost();

};

struct PathSeerStats {
    size_t n1, n2, n3;
    size_t ndis, ndis_first;
    size_t nfilter, nfilter_first;
    size_t nhop, nhop_first;
    size_t virtual_overhead;
    
    PathSeerStats(
            size_t n1 = 0,
            size_t n2 = 0,
            size_t n3 = 0,
            size_t ndis = 0,
            size_t ndis_first = 0,
            size_t nfilter = 0,
            size_t nfilter_first = 0,
            size_t nhop = 0,
            size_t nhop_first = 0,
            size_t virtual_overhead = 0
            )
            : n1(n1), n2(n2), n3(n3), ndis(ndis), ndis_first(ndis_first), nfilter(nfilter), nfilter_first(nfilter_first), nhop(nhop), nhop_first(nhop_first), virtual_overhead(virtual_overhead) {}

    void reset() {
        n1 = n2 = n3 = 0;
        ndis = 0;
        ndis_first = 0;
        nhop = 0;
        nhop_first = 0;
        nfilter = 0;
        nfilter_first = 0;
        virtual_overhead = 0;
    }

    void combine(const PathSeerStats& other) {
        n1 += other.n1;
        n2 += other.n2;
        n3 += other.n3;
        ndis += other.ndis;
        nhop += other.nhop;
        nhop_first += other.nhop_first;
        ndis_first += other.ndis_first;
        nfilter += other.nfilter;
        nfilter_first += other.nfilter_first;
        virtual_overhead += other.virtual_overhead;
    }
};

// global var that collects them all
FAISS_API extern PathSeerStats pathseer_stats;

} // namespace faiss
