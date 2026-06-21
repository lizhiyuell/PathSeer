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

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>

namespace faiss {

/** Implementation of the Hierarchical Navigable Small World
 * datastructure.
 *
 * Efficient and robust approximate nearest neighbor search using
 * Hierarchical Navigable Small World graphs
 *
 *  Yu. A. Malkov, D. A. Yashunin, arXiv 2017
 *
 * This implementation is heavily influenced by the NMSlib
 * implementation by Yury Malkov and Leonid Boystov
 * (https://github.com/searchivarius/nmslib)
 *
 * The NaviX object stores only the neighbor link structure, see
 * IndexNaviX.h for the full index object.
 */

struct VisitedTable;
struct DistanceComputer; // from AuxIndexStructures
struct NaviXStats;

template <class C>
struct ResultHandler;//lbj+

struct SearchParametersNaviX : SearchParameters {
    int efSearch = 16;
    bool check_relative_distance = true;
    bool bounded_queue = true;//lbj+?
    bool filter_opt = false;

    ~SearchParametersNaviX() {}
};

struct NaviX {
    /// internal storage of vectors (32 bits: this is expensive)
    using storage_idx_t = int32_t;

    // for now we do only these distances
    using C = CMax<float, int64_t>;

    typedef std::pair<float, storage_idx_t> Node;

    typedef std::array<storage_idx_t, 4096> node_array_t;

    /** Heap structure that allows fast
     */
    struct MinimaxHeap {
        int n;
        int k;
        int nvalid;

        std::vector<storage_idx_t> ids;
        std::vector<float> dis;
        typedef faiss::CMax<float, storage_idx_t> HC;

        explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0), ids(n), dis(n) {}

        void push(storage_idx_t i, float v);

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
        bool operator<(const NodeDistCloser& obj1) const {
            return d < obj1.d;
        }
    };

    struct NodeDistFarther {
        float d;
        int id;
        NodeDistFarther(float d, int id) : d(d), id(id) {}
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
    std::vector<storage_idx_t> neighbors;

    /// entry point in the search structure (one of the points with maximum
    /// level
    storage_idx_t entry_point = -1;

    faiss::RandomGenerator rng;

    /// maximum level
    int max_level = -1;

    /// expansion factor at construction time
    int efConstruction = 40;

    /// expansion factor at search time
    int efSearch = 20;

    /// during search: do we check whether the next best distance is good
    /// enough?
    bool check_relative_distance = true;

    /// number of entry points in levels > 0.
    int upper_beam = 1;

    /// use bounded queue during exploration
    bool search_bounded_queue = true;

    // methods that initialize the tree sizes

    /// initialize the assign_probas and cum_nneighbor_per_level to
    /// have 2*M links on level 0 and M links on levels > 0
    void set_default_probas(int M, float levelMult);

    /// set nb of neighbors for this level (before adding anything)
    void set_nb_neighbors(int level_no, int n);

    // methods that access the tree sizes

    /// nb of neighbors for this level
    int nb_neighbors(int layer_no) const;

    /// cumumlative nb up to (and excluding) this level
    int cum_nb_neighbors(int layer_no) const;

    /// range of entries in the neighbors table of vertex no at layer_no
    void neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
            const;

    /// only mandatory parameter: nb of neighbors
    explicit NaviX(int M = 32);

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

    /** add point pt_id on all levels <= pt_level and build the link
     * structure for them. */
    void add_with_locks(
            DistanceComputer& ptdis,
            int pt_level,
            int pt_id,
            std::vector<omp_lock_t>& locks,
            VisitedTable& vt);

    /// search interface for 1 point, single thread
    NaviXStats search(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            VisitedTable& vt,
            int pos = 0,
            const SearchParametersNaviX* params = nullptr) const;

    //lbj+
    NaviXStats search(
        DistanceComputer& qdis,
        ResultHandler<C>& res,
        VisitedTable& vt,
        const SearchParameters* params = nullptr) const;

    /// Navix Hybrid Search!!!
    inline void navix_push_to_results(int k, float *D, idx_t *I, float dist, idx_t v, int &nres) const {
        if (nres < k) {
            faiss::maxheap_push(++nres, D, I, dist, v);
        } else if (dist < D[0]) {
            faiss::maxheap_replace_top(nres, D, I, dist, v);
        }
    }

    inline int navix_batch_compute_distance(
        node_array_t &node_array,
        int &size,
        DistanceComputer &qdis,
        MinimaxHeap &candidates,
        ResultHandler<C> &res,
        NaviXStats &stats) const;

    inline void navix_one_hop(
        size_t begin,
        size_t end,
        VisitedTable &vt,
        const IDSelector* sel,
        int query_id,
        node_array_t &node_array,
        int &size,
        NaviXStats &stat) const;

    inline void navix_one_hop_filter_opt(
        size_t begin,
        size_t end,
        VisitedTable &vt,
        const IDSelector* sel,
        int query_id,
        node_array_t &node_array,
        int &size,
        NaviXStats &stat) const;

    inline int navix_batch_directed_compute_distance(
        node_array_t &node_array,
        int &size,
        // const char *filter_id_map,//lbj+
        const IDSelector* sel,
        int query_id,
        DistanceComputer &qdis,
        std::priority_queue<NodeDistFarther> &nbrs_to_explore,
        MinimaxHeap &candidates,
        ResultHandler<C> &res,
        NaviXStats &stats) const;

    inline int navix_batch_directed_compute_distance_filter_opt(
        node_array_t &node_array,
        int &size,
        // const char *filter_id_map,//lbj+
        const IDSelector* sel,
        int query_id,
        DistanceComputer &qdis,
        std::priority_queue<NodeDistFarther> &nbrs_to_explore,
        MinimaxHeap &candidates,
        ResultHandler<C> &res,
        NaviXStats &stats,
        VisitedTable &vt) const;

    inline int navix_directed(
        size_t begin,
        size_t end,
        DistanceComputer &qdis,
        MinimaxHeap &candidates,
        ResultHandler<C> &res,
        VisitedTable &vt,
        const IDSelector* sel,
        int query_id,
        int filter_nbrs_to_find,
        node_array_t &node_array,
        int &size,
        NaviXStats &stats) const;

    inline int navix_directed_filter_opt(
        size_t begin,
        size_t end,
        DistanceComputer &qdis,
        MinimaxHeap &candidates,
        ResultHandler<C> &res,
        VisitedTable &vt,
        const IDSelector* sel,
        int query_id,
        int filter_nbrs_to_find,
        node_array_t &node_array,
        int &size,
        NaviXStats &stats) const;

    inline void navix_blind(
        size_t begin,
        size_t end,
        VisitedTable &vt,
        const char *filter_id_map,
        int filter_nbrs_to_find,
        node_array_t &node_array,
        int &size,
        NaviXStats &stats) const;

    inline void navix_full_two_hop(
        size_t begin,
        size_t end,
        VisitedTable &vt,
        const IDSelector* sel,
        int query_id,
        node_array_t &node_array,
        int &size,
        NaviXStats &stats) const;

    inline void navix_full_two_hop_filter_opt(
        size_t begin,
        size_t end,
        VisitedTable &vt,
        const IDSelector* sel,
        int query_id,
        node_array_t &node_array,
        int &size,
        NaviXStats &stats) const;

    inline int navix_add_filtered_nodes_to_candidates(
        DistanceComputer &qdis,
        MinimaxHeap &candidates,
        ResultHandler<C> &res,
        VisitedTable &vt,
        const IDSelector* sel,
        int query_id,
        int num_of_nodes,
        NaviXStats &stats)const;

    inline int navix_add_filtered_nodes_to_candidates_filter_opt(
        DistanceComputer &qdis,
        MinimaxHeap &candidates,
        ResultHandler<C> &res,
        VisitedTable &vt,
        const IDSelector* sel,
        int query_id,
        int num_of_nodes,
        NaviXStats &stats)const;

    NaviXStats navix_hybrid_search(
            DistanceComputer &qdis,
            ResultHandler<C> &res,
            VisitedTable &vt,
            const IDSelector* sel,
            int query_id,
            const SearchParametersNaviX *params) const;

    NaviXStats navix_hybrid_search_filter_opt(
            DistanceComputer &qdis,
            ResultHandler<C> &res,
            VisitedTable &vt,
            const IDSelector* sel,
            int query_id,
            const SearchParametersNaviX *params) const;

        /// search only in level 0 from a given vertex
    void search_level_0(
            DistanceComputer& qdis,
            ResultHandler<C>& res,
            idx_t nprobe,
            const storage_idx_t* nearest_i,
            const float* nearest_d,
            int search_type,
            NaviXStats& search_stats,
            VisitedTable& vt,
            const SearchParameters* params = nullptr) const;
    //lbj-

    void reset();

    void clear_neighbor_tables(int level);
    void print_neighbor_stats(int level) const;

    int prepare_level_tab(size_t n, bool preset_levels = false);

    static void shrink_neighbor_list(
            DistanceComputer& qdis,
            std::priority_queue<NodeDistFarther>& input,
            std::vector<NodeDistFarther>& output,
            int max_size);
    
    void permute_entries(const idx_t* map);//lbj+
};

// we modify the NaviXStats to count for more operations
struct NaviXStats {
    size_t n1, n2, n3;
    size_t ndis, ndis_first;
    size_t nfilter, nfilter_first;
    size_t nhops, nhop_first;
    size_t ngood, ngood_first;
    size_t nreorder;
    float useful_rate;

    NaviXStats(
            size_t n1 = 0,
            size_t n2 = 0,
            size_t n3 = 0,
            size_t ndis = 0,
            size_t ndis_first = 0,
            size_t nfilter = 0,
            size_t nfilter_first = 0,
            size_t nhops = 0,
            size_t nhop_first = 0,
            size_t ngood = 0,
            size_t ngood_first = 0,
            size_t nreorder = 0,
            float useful_rate = 0)
            : n1(n1), n2(n2), n3(n3), ndis(ndis), ndis_first(ndis_first), nfilter(nfilter), nfilter_first(nfilter_first), nhops(nhops), nhop_first(nhop_first), ngood(ngood), ngood_first(ngood_first) , nreorder(nreorder), useful_rate(useful_rate) {}

    void reset() {
        n1 = n2 = n3 = 0;
        ndis = 0;
        ndis_first = 0;
        nhops = 0;
        nhop_first = 0;
        nfilter = 0;
        nfilter_first = 0;
        ngood = 0;
        ngood_first = 0;
        nreorder = 0;
        useful_rate = 0;
    }

    void combine(const NaviXStats& other) {
        n1 += other.n1;
        n2 += other.n2;
        n3 += other.n3;
        ndis += other.ndis;
        nhops += other.nhops;
        nhop_first += other.nhop_first;
        ndis_first += other.ndis_first;
        nfilter += other.nfilter;
        nfilter_first += other.nfilter_first;
        ngood += other.ngood;
        ngood_first += other.ngood_first;
        nreorder += other.nreorder;
        useful_rate += other.useful_rate;
    }
};

// global var that collects them all
FAISS_API extern NaviXStats navix_stats;

} // namespace faiss
