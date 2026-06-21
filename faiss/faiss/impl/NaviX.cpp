/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*- 

#include <faiss/impl/NaviX.h>

#include <string>

#ifdef __AVX2__
#include <immintrin.h>

#include <limits>
#include <type_traits>
#endif

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>//lbj+
#include <faiss/utils/prefetch.h>

static inline bool get_and_set_member(const faiss::IDSelector*sel, faiss::VisitedTable& vt, int64_t v, int& nr_filter, int pos){

    uint8_t version = vt.visited[v];
    uint8_t visno = vt.visno;
    if(version >= vt.visno-1)
        return version==visno;
    else{
        bool res = sel->is_member(v, pos);
        nr_filter++;
        vt.visited[v] = visno - 1 + res;
        return res;
    }

}

namespace faiss {

/**************************************************************
 * NaviX structure implementation
 **************************************************************/

int NaviX::nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no + 1] -
            cum_nneighbor_per_level[layer_no];
}

void NaviX::set_nb_neighbors(int level_no, int n) {
    FAISS_THROW_IF_NOT(levels.size() == 0);
    int cur_n = nb_neighbors(level_no);
    for (int i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
        cum_nneighbor_per_level[i] += n - cur_n;
    }
}

int NaviX::cum_nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no];
}

void NaviX::neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
        const {
    size_t o = offsets[no];
    *begin = o + cum_nb_neighbors(layer_no);
    *end = o + cum_nb_neighbors(layer_no + 1);
}

NaviX::NaviX(int M) : rng(12345) {
    set_default_probas(M, 1.0 / log(M));
    offsets.push_back(0);
}

int NaviX::random_level() {
    double f = rng.rand_float();
    // could be a bit faster with bissection
    for (int level = 0; level < assign_probas.size(); level++) {
        if (f < assign_probas[level]) {
            return level;
        }
        f -= assign_probas[level];
    }
    // happens with exponentially low probability
    return assign_probas.size() - 1;
}

void NaviX::set_default_probas(int M, float levelMult) {
    int nn = 0;
    cum_nneighbor_per_level.push_back(0);
    for (int level = 0;; level++) {
        float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
        if (proba < 1e-9)
            break;
        assign_probas.push_back(proba);
        nn += level == 0 ? M * 2 : M;
        cum_nneighbor_per_level.push_back(nn);
    }
}

void NaviX::clear_neighbor_tables(int level) {
    for (int i = 0; i < levels.size(); i++) {
        size_t begin, end;
        neighbor_range(i, level, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            neighbors[j] = -1;
        }
    }
}

void NaviX::reset() {
    max_level = -1;
    entry_point = -1;
    offsets.clear();
    offsets.push_back(0);
    levels.clear();
    neighbors.clear();
}

void NaviX::print_neighbor_stats(int level) const {
    FAISS_THROW_IF_NOT(level < cum_nneighbor_per_level.size());
    printf("stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));
    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
  reduction(+: tot_reciprocal) reduction(+: n_node)
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] < 0)
                    break;
                neighset.insert(neighbors[j]);
            }
            int n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j];
                if (i2 < 0)
                    break;
                FAISS_ASSERT(i2 != i);
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = neighbors[j2];
                    if (i3 < 0)
                        break;
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;
    printf("   nb of nodes at that level %zd\n", n_node);
    printf("   neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);
}

void NaviX::fill_with_random_links(size_t n) {
    int max_level = prepare_level_tab(n);
    RandomGenerator rng2(456);

    for (int level = max_level - 1; level >= 0; --level) {
        std::vector<int> elts;
        for (int i = 0; i < n; i++) {
            if (levels[i] > level) {
                elts.push_back(i);
            }
        }
        printf("linking %zd elements in level %d\n", elts.size(), level);

        if (elts.size() == 1)
            continue;

        for (int ii = 0; ii < elts.size(); ii++) {
            int i = elts[ii];
            size_t begin, end;
            neighbor_range(i, 0, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                int other = 0;
                do {
                    other = elts[rng2.rand_int(elts.size())];
                } while (other == i);

                neighbors[j] = other;
            }
        }
    }
}

int NaviX::prepare_level_tab(size_t n, bool preset_levels) {
    size_t n0 = offsets.size() - 1;

    if (preset_levels) {
        FAISS_ASSERT(n0 + n == levels.size());
    } else {
        FAISS_ASSERT(n0 == levels.size());
        for (int i = 0; i < n; i++) {
            int pt_level = random_level();
            levels.push_back(pt_level + 1);
        }
    }

    int max_level = 0;
    for (int i = 0; i < n; i++) {
        int pt_level = levels[i + n0] - 1;
        if (pt_level > max_level)
            max_level = pt_level;
        offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
        neighbors.resize(offsets.back(), -1);
    }

    return max_level;
}

/** Enumerate vertices from farthest to nearest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
void NaviX::shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistFarther>& input,
        std::vector<NodeDistFarther>& output,
        int max_size) {
    while (input.size() > 0) {
        NodeDistFarther v1 = input.top();
        input.pop();
        float dist_v1_q = v1.d;

        bool good = true;
        for (NodeDistFarther v2 : output) {
            float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);

            if (dist_v1_v2 < dist_v1_q) {
                good = false;
                break;
            }
        }

        if (good) {
            output.push_back(v1);
            if (output.size() >= max_size) {
                return;
            }
        }
    }
}

namespace {

using storage_idx_t = NaviX::storage_idx_t;
using NodeDistCloser = NaviX::NodeDistCloser;
using NodeDistFarther = NaviX::NodeDistFarther;

/**************************************************************
 * Addition subroutines
 **************************************************************/

/// remove neighbors from the list to make it smaller than max_size
void shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& resultSet1,
        int max_size) {
    if (resultSet1.size() < max_size) {
        return;
    }
    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    NaviX::shrink_neighbor_list(qdis, resultSet, returnlist, max_size);

    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }
}

/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
void add_link(
        NaviX& navix,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level) {
    size_t begin, end;
    navix.neighbor_range(src, level, &begin, &end);
    if (navix.neighbors[end - 1] == -1) {
        // there is enough room, find a slot to add it
        size_t i = end;
        while (i > begin) {
            if (navix.neighbors[i - 1] != -1)
                break;
            i--;
        }
        navix.neighbors[i] = dest;
        return;
    }

    // otherwise we let them fight out which to keep

    // copy to resultSet...
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) { // HERE WAS THE BUG
        storage_idx_t neigh = navix.neighbors[i];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    shrink_neighbor_list(qdis, resultSet, end - begin);

    // ...and back
    size_t i = begin;
    while (resultSet.size()) {
        navix.neighbors[i++] = resultSet.top().id;
        resultSet.pop();
    }
    // they may have shrunk more than just by 1 element
    while (i < end) {
        navix.neighbors[i++] = -1;
    }
}

/// search neighbors on a single level, starting from an entry point
void search_neighbors_to_add(
        NaviX& navix,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt) {
    // top is nearest candidate
    std::priority_queue<NodeDistFarther> candidates;

    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    while (!candidates.empty()) {
        // get nearest
        const NodeDistFarther& currEv = candidates.top();

        if (currEv.d > results.top().d) {
            break;
        }
        int currNode = currEv.id;
        candidates.pop();

        // loop over neighbors
        size_t begin, end;
        navix.neighbor_range(currNode, level, &begin, &end);
        for (size_t i = begin; i < end; i++) {
            storage_idx_t nodeId = navix.neighbors[i];
            if (nodeId < 0)
                break;
            if (vt.get(nodeId))
                continue;
            vt.set(nodeId);

            float dis = qdis(nodeId);
            NodeDistFarther evE1(dis, nodeId);

            if (results.size() < navix.efConstruction || results.top().d > dis) {
                results.emplace(dis, nodeId);
                candidates.emplace(dis, nodeId);
                if (results.size() > navix.efConstruction) {
                    results.pop();
                }
            }
        }
    }
    vt.advance();
}

/**************************************************************
 * Searching subroutines
 **************************************************************/

/// greedily update a nearest vector at a given level
NaviXStats greedy_update_nearest(
        const NaviX& navix,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {

    NaviXStats stats;
    int ndis = 0;
    int nhops = 0;

    for (;;) {
        storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        navix.neighbor_range(nearest, level, &begin, &end);
        for (size_t i = begin; i < end; i++) {
            storage_idx_t v = navix.neighbors[i];
            if (v < 0)
                break;
            ndis++;
            float dis = qdis(v);
            #ifdef OUTPUT_SEARCH_PATH
            printf("%d\t%f\t%d\n", v, dis, 0);
            #endif
            if (dis < d_nearest) {
                nearest = v;
                d_nearest = dis;
            }
        }

        nhops++;

        if (nearest == prev_nearest) {
            stats.ndis = ndis;
            stats.ndis_first = ndis;
            stats.nhops = nhops;
            stats.nhop_first = nhops;
            return stats;
        }
    }
}

} // namespace

/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
void NaviX::add_links_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        omp_lock_t* locks,
        VisitedTable& vt) {
    std::priority_queue<NodeDistCloser> link_targets;

    search_neighbors_to_add(
            *this, ptdis, link_targets, nearest, d_nearest, level, vt);

    // but we can afford only this many neighbors
    int M = nb_neighbors(level);

    ::faiss::shrink_neighbor_list(ptdis, link_targets, M);

    std::vector<storage_idx_t> neighbors;
    neighbors.reserve(link_targets.size());
    while (!link_targets.empty()) {
        storage_idx_t other_id = link_targets.top().id;
        add_link(*this, ptdis, pt_id, other_id, level);
        neighbors.push_back(other_id);
        link_targets.pop();
    }

    omp_unset_lock(&locks[pt_id]);
    for (storage_idx_t other_id : neighbors) {
        omp_set_lock(&locks[other_id]);
        add_link(*this, ptdis, other_id, pt_id, level);
        omp_unset_lock(&locks[other_id]);
    }
    omp_set_lock(&locks[pt_id]);
}

/**************************************************************
 * Building, parallel
 **************************************************************/

void NaviX::add_with_locks(
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt) {
    //  greedy search on upper levels

    storage_idx_t nearest;
#pragma omp critical
    {
        nearest = entry_point;

        if (nearest == -1) {
            max_level = pt_level;
            entry_point = pt_id;
        }
    }

    if (nearest < 0) {
        return;
    }

    omp_set_lock(&locks[pt_id]);

    int level = max_level; // level at which we start adding neighbors
    float d_nearest = ptdis(nearest);

    for (; level > pt_level; level--) {
        greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
    }

    for (; level >= 0; level--) {
        add_links_starting_from(
                ptdis, pt_id, nearest, d_nearest, level, locks.data(), vt);
    }

    omp_unset_lock(&locks[pt_id]);

    if (pt_level > max_level) {
        max_level = pt_level;
        entry_point = pt_id;
    }
}

/**************************************************************
 * Searching
 **************************************************************/

namespace {

using MinimaxHeap = NaviX::MinimaxHeap;
using Node = NaviX::Node;
using C = NaviX::C;//lbj+
// just used as a lower bound for the minmaxheap, but it is set for heap search
int extract_k_from_ResultHandler(ResultHandler<C>& res) {
    using RH = HeapBlockResultHandler<C>;
    if (auto hres = dynamic_cast<RH::SingleResultHandler*>(&res)) {
        return hres->k;
    }
    return 1;
}//lbj+

// /** Do a BFS on the candidates list */

int search_from_candidates(
        const NaviX& navix,
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        NaviXStats& stats,
        int level,
        int pos,
        int nres_in = 0,
        const SearchParametersNaviX* params = nullptr) {
    int nres = nres_in;
    int ndis = 0;
    int nfilter = 0;
    int ngood = 0;

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : navix.check_relative_distance;
    int efSearch = params ? params->efSearch : navix.efSearch;
    const IDSelector* sel = params ? params->sel : nullptr;

    #ifdef COUNT_USEFUL_RATES
    std::priority_queue<float> count_useful_queue;
    #endif


    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        nfilter++;
        if (!sel || sel->is_member(v1, pos)) {
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
                ngood++;
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
                ngood++;
            }
        }
        vt.set(v1);
        #ifdef COUNT_USEFUL_RATES
        count_useful_queue.push(d);
        #endif
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        size_t begin, end;
        navix.neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = navix.neighbors[j];
            if (v1 < 0)
                break;
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);
            ndis++;
            float d = qdis(v1);
            #ifdef OUTPUT_SEARCH_PATH
            printf("%d\t%f\t%d\n", v1, d, sel->is_member(v1, pos)?2:3);
            #endif
            #ifdef COUNT_USEFUL_RATES
            count_useful_queue.push(d);
            #endif
            if (nres < k || d < D[0]){
                nfilter++;
                if (!sel || sel->is_member(v1, pos)) {
                    // printf("[One for res]\n");
                    if (nres < k) {
                        faiss::maxheap_push(++nres, D, I, d, v1);
                        ngood++;
                    } else if (d < D[0]) {
                        faiss::maxheap_replace_top(nres, D, I, d, v1);
                        ngood++;
                    }
                }
            }
            candidates.push(v1, d);

        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        stats.nhops += nstep;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nfilter += nfilter;
        stats.ngood += ngood;
        #ifdef COUNT_USEFUL_RATES
        float dist_threshold = D[0];
        size_t nr_total = count_useful_queue.size();
        while(count_useful_queue.top() > dist_threshold)
            count_useful_queue.pop();
        size_t nr_useful = count_useful_queue.size();
        float useful_rate = nr_total ? 1.0 * nr_useful / nr_total : 0;
        stats.useful_rate = useful_rate;
        #endif
    }

    return nres;
}//lbj+old， below is new

/** Do a BFS on the candidates list */
int search_from_candidates(
        const NaviX& navix,
        DistanceComputer& qdis,
        ResultHandler<C>& res,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        NaviXStats& stats,
        int level,
        int nres_in,
        const SearchParameters* params) {
    int nres = nres_in;
    int ndis = 0;

    // can be overridden by search params
    bool do_dis_check = navix.check_relative_distance;
    int efSearch = navix.efSearch;
    const IDSelector* sel = nullptr;
    if (params) {
        if (const SearchParametersNaviX* navix_params =
                    dynamic_cast<const SearchParametersNaviX*>(params)) {
            do_dis_check = navix_params->check_relative_distance;
            efSearch = navix_params->efSearch;
        }
        sel = params->sel;
    }

    C::T threshold = res.threshold;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (d < threshold) {
                if (res.add_result(d, v1)) {
                    threshold = res.threshold;
                }
            }
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        size_t begin, end;
        navix.neighbor_range(v0, level, &begin, &end);

        // a faster version: reference version in unit test test_navix.cpp
        // the following version processes 4 neighbors at a time
        size_t jmax = begin;
        for (size_t j = begin; j < end; j++) {
            int v1 = navix.neighbors[j];
            if (v1 < 0)
                break;

            prefetch_L2(vt.visited.data() + v1);
            jmax += 1;
        }

        int counter = 0;
        size_t saved_j[4];

        threshold = res.threshold;

        auto add_to_heap = [&](const size_t idx, const float dis) {
            if (!sel || sel->is_member(idx)) {
                if (dis < threshold) {
                    if (res.add_result(dis, idx)) {
                        threshold = res.threshold;
                        nres += 1;
                    }
                }
            }
            candidates.push(idx, dis);
        };

        for (size_t j = begin; j < jmax; j++) {
            int v1 = navix.neighbors[j];

            bool vget = vt.get(v1);
            vt.set(v1);
            saved_j[counter] = v1;
            counter += vget ? 0 : 1;

            if (counter == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        saved_j[0],
                        saved_j[1],
                        saved_j[2],
                        saved_j[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    add_to_heap(saved_j[id4], dis[id4]);
                }

                ndis += 4;

                counter = 0;
            }
        }

        for (size_t icnt = 0; icnt < counter; icnt++) {
            float dis = qdis(saved_j[icnt]);
            add_to_heap(saved_j[icnt], dis);

            ndis += 1;
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nhops += nstep;
    }

    return nres;
}//lbj+new

std::priority_queue<NaviX::Node> search_from_candidate_unbounded(
        const NaviX& navix,
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        NaviXStats& stats) {
    int ndis = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);

    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }

        candidates.pop();

        size_t begin, end;
        navix.neighbor_range(v0, 0, &begin, &end);

        for (size_t j = begin; j < end; ++j) {
            int v1 = navix.neighbors[j];

            if (v1 < 0) {
                break;
            }
            if (vt->get(v1)) {
                continue;
            }

            vt->set(v1);

            float d1 = qdis(v1);
            ++ndis;

            if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                candidates.emplace(d1, v1);
                top_candidates.emplace(d1, v1);

                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        }
    }

    ++stats.n1;
    if (candidates.size() == 0) {
        ++stats.n2;
    }
    stats.n3 += ndis;

    return top_candidates;
}

} // anonymous namespace

NaviXStats NaviX::search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt,
        int pos,
        const SearchParametersNaviX* params) const {
    NaviXStats stats;
    if (entry_point == -1) {
        return stats;
    }
    if (upper_beam == 1) {
        //  greedy search on upper levels
        storage_idx_t nearest = entry_point;
        float d_nearest = qdis(nearest);

        for (int level = max_level; level >= 1; level--) {
            NaviXStats local_stats = greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
            stats.combine(local_stats);
        }

        int ef = std::max(params ? params->efSearch : efSearch, k);
        if (search_bounded_queue) { // this is the most common branch
            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);

            search_from_candidates(
                    *this, qdis, k, I, D, candidates, vt, stats, 0, pos, 0, params);
        } else {
            std::priority_queue<Node> top_candidates =
                    search_from_candidate_unbounded(
                            *this,
                            Node(d_nearest, nearest),
                            qdis,
                            ef,
                            &vt,
                            stats);

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }

            int nres = 0;
            while (!top_candidates.empty()) {
                float d;
                storage_idx_t label;
                std::tie(d, label) = top_candidates.top();
                faiss::maxheap_push(++nres, D, I, d, label);
                top_candidates.pop();
            }
        }

        vt.advance();

    } else {
        int candidates_size = upper_beam;
        MinimaxHeap candidates(candidates_size);

        std::vector<idx_t> I_to_next(candidates_size);
        std::vector<float> D_to_next(candidates_size);

        int nres = 1;
        I_to_next[0] = entry_point;
        D_to_next[0] = qdis(entry_point);

        for (int level = max_level; level >= 0; level--) {
            // copy I, D -> candidates

            candidates.clear();

            for (int i = 0; i < nres; i++) {
                candidates.push(I_to_next[i], D_to_next[i]);
            }

            if (level == 0) {
                nres = search_from_candidates(
                        *this, qdis, k, I, D, candidates, vt, stats, pos, 0);
            } else {
                nres = search_from_candidates(
                        *this,
                        qdis,
                        candidates_size,
                        I_to_next.data(),
                        D_to_next.data(),
                        candidates,
                        vt,
                        stats,
                        level,
                        pos);
            }
            vt.advance();
        }
    }

    return stats;
}

//lbj+
NaviXStats NaviX::search(
        DistanceComputer& qdis,
        ResultHandler<C>& res,
        VisitedTable& vt,
        const SearchParameters* params) const {
    NaviXStats stats;
    if (entry_point == -1) {
        return stats;
    }
    int k = extract_k_from_ResultHandler(res);

    bool bounded_queue = this->search_bounded_queue;
    int efSearch = this->efSearch;
    if (params) {
        if (const SearchParametersNaviX* navix_params =
                    dynamic_cast<const SearchParametersNaviX*>(params)) {
            bounded_queue = navix_params->bounded_queue;
            efSearch = navix_params->efSearch;
        }
    }

    //  greedy search on upper levels
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for (int level = max_level; level >= 1; level--) {
        NaviXStats local_stats =
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        stats.combine(local_stats);
    }

    int ef = std::max(efSearch, k);
    if (bounded_queue) { // this is the most common branch
        MinimaxHeap candidates(ef);

        candidates.push(nearest, d_nearest);

        search_from_candidates(
                *this, qdis, res, candidates, vt, stats, 0, 0, params);
    } else {
        std::priority_queue<Node> top_candidates =
                search_from_candidate_unbounded(
                        *this, Node(d_nearest, nearest), qdis, ef, &vt, stats);

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }

        while (!top_candidates.empty()) {
            float d;
            storage_idx_t label;
            std::tie(d, label) = top_candidates.top();
            res.add_result(d, label);
            top_candidates.pop();
        }
    }

    vt.advance();

    return stats;
}

/// Navix search methods
int NaviX::navix_batch_compute_distance(
    node_array_t &node_array,
    int &size,
    DistanceComputer &qdis,
    MinimaxHeap &candidates,
    ResultHandler<C> &res,
    NaviXStats &stats) const {
    auto threshold = res.threshold;
    auto nres = 0;
    auto add_to_heap = [&](const storage_idx_t idx, const float dis) {
        if (dis < threshold) {
            if (res.add_result(dis, idx)) {
                threshold = res.threshold;
                nres += 1;
            }
        }
        candidates.push(idx, dis);
    };

    for (int i=0; i < size; i++) {
        const float dist = qdis(node_array[i]);
        stats.ndis += 1;
        add_to_heap(node_array[i], dist);
    }

    // Reset the size
    size = 0;
    return nres;
}


void NaviX::navix_one_hop(
    size_t begin,
    size_t end,
    VisitedTable &vt,
    const IDSelector *sel,
    int query_id,
    node_array_t &node_array,
    int &size,
    NaviXStats &stat) const {
    int nfilter = 0;
    for (size_t j = begin; j < end; ++j) {
        int v1 = neighbors[j];
        if (vt.get(v1)) continue;
        nfilter++;
        if (sel->is_member(v1, query_id)) {
            if (size < node_array.size()) node_array[size++] = v1;
            vt.set(v1);
        }
    }
    stat.nfilter += nfilter;
}

void NaviX::navix_one_hop_filter_opt(
    size_t begin,
    size_t end,
    VisitedTable &vt,
    const IDSelector *sel,
    int query_id,
    node_array_t &node_array,
    int &size,
    NaviXStats &stat) const {
    int nfilter = 0;
    for (size_t j = begin; j < end; ++j) {
        int v1 = neighbors[j];
        if (vt.get(v1)) continue;
        if (get_and_set_member(sel, vt, v1, nfilter, query_id)) {
            if (size < node_array.size()) node_array[size++] = v1;
            vt.set(v1);
        }
    }
    stat.nfilter += nfilter;
}

int NaviX::navix_batch_directed_compute_distance(
    node_array_t &node_array,
    int &size,
    const IDSelector *sel,
    int query_id,
    DistanceComputer &qdis,
    std::priority_queue<NodeDistFarther> &nbrs_to_explore,
    MinimaxHeap &candidates,
    ResultHandler<C> &res,
    NaviXStats &stats) const {
    auto threshold = res.threshold;
    auto nres = 0;

    for (int i=0; i < size; i++) {
        const float dist = qdis(node_array[i]);
        stats.ndis++;
        nbrs_to_explore.emplace(dist, node_array[i]);
        
        stats.nfilter++;
        if (!sel->is_member(node_array[i], query_id)) {
            continue;
        }

        if (dist < threshold) {
            if (res.add_result(dist, node_array[i])) {
                threshold = res.threshold;
                nres += 1;
            }
        }
        candidates.push(node_array[i], dist);
    }

    // Reset the size
    size = 0;
    return nres;
}

int NaviX::navix_batch_directed_compute_distance_filter_opt(
    node_array_t &node_array,
    int &size,
    const IDSelector *sel,
    int query_id,
    DistanceComputer &qdis,
    std::priority_queue<NodeDistFarther> &nbrs_to_explore,
    MinimaxHeap &candidates,
    ResultHandler<C> &res,
    NaviXStats &stats,
    VisitedTable &vt) const {
    auto threshold = res.threshold;
    auto nres = 0;

    int nfilter = 0;

    for (int i=0; i < size; i++) {
        const float dist = qdis(node_array[i]);
        stats.ndis++;
        nbrs_to_explore.emplace(dist, node_array[i]);
        
        nfilter++;
        if (!get_and_set_member(sel, vt, node_array[i], nfilter, query_id)) {
            continue;
        }

        if (dist < threshold) {
            if (res.add_result(dist, node_array[i])) {
                threshold = res.threshold;
                nres += 1;
            }
        }
        candidates.push(node_array[i], dist);
    }

    // Reset the size
    size = 0;
    stats.nfilter += nfilter;

    return nres;
}

int NaviX::navix_directed(
    size_t begin,
    size_t end,
    DistanceComputer &qdis,
    MinimaxHeap &candidates,
    ResultHandler<C> &res,
    VisitedTable &vt,
    const IDSelector *sel,
    int query_id,
    int filter_nbrs_to_find,
    node_array_t &node_array,
    int &size,
    NaviXStats &stats) const {
    std::priority_queue<NodeDistFarther> nbrs_to_explore;

    int nres = 0;
    int nfilter = 0;

    // First Hop Neighbors
    int visited_set_size = 0;
    for (size_t j = begin; j < end; ++j) {
        auto v1 = neighbors[j];
        nfilter++;
        bool is_masked = (sel->is_member(v1, query_id));
        if (is_masked) {
            visited_set_size++;
        }

        if (vt.get(v1)) {
            continue;
        }

        if (is_masked) {
            vt.set(v1);
        }
        node_array[size++] = v1;

    }

    // add the first-hop neighbors to result/candidate heap, as well as sorting them to find the suitable two-hop neighbors
    nres += navix_batch_directed_compute_distance(
        node_array,
        size,
        sel,
        query_id,
        qdis,
        nbrs_to_explore,
        candidates,
        res,
        stats);

    while (!nbrs_to_explore.empty()) {
        auto nbrs = nbrs_to_explore.top();
        nbrs_to_explore.pop();

        if (visited_set_size >= filter_nbrs_to_find) {
            break;
        }
        if (vt.get(nbrs.id)) {
            continue;
        }
        vt.set(nbrs.id);

        size_t second_begin, second_end;
        neighbor_range(nbrs.id, 0, &second_begin, &second_end);
        stats.nhops += 1;

        for (size_t j = second_begin; j < second_end; ++j) {
            int v2 = neighbors[j];
            if (v2 < 0) {
                second_end = j;
                break;
            }
            prefetch_L2(vt.visited.data() + v2);
        }

        for (size_t j = second_begin; j < second_end; ++j) {
            auto v2 = neighbors[j];
            nfilter++;
            bool filter_mask = (sel->is_member(v2, query_id));
            if (filter_mask) {
                visited_set_size++;
            }
            if (vt.get(v2)) {
                continue;
            }
            if (filter_mask) {
                if (size < node_array.size()) node_array[size++] = v2;
                vt.set(v2);
            }
        }
    }
    stats.nfilter += nfilter;
    return nres;
}

int NaviX::navix_directed_filter_opt(
    size_t begin,
    size_t end,
    DistanceComputer &qdis,
    MinimaxHeap &candidates,
    ResultHandler<C> &res,
    VisitedTable &vt,
    const IDSelector *sel,
    int query_id,
    int filter_nbrs_to_find,
    node_array_t &node_array,
    int &size,
    NaviXStats &stats) const {
    std::priority_queue<NodeDistFarther> nbrs_to_explore;

    int nres = 0;
    int nfilter = 0;

    // First Hop Neighbors
    int visited_set_size = 0;
    for (size_t j = begin; j < end; ++j) {
        auto v1 = neighbors[j];
        nfilter++;
        bool prev_visited = vt.get_either(v1);
        bool is_masked = (get_and_set_member(sel, vt, v1, nfilter, query_id));
        if (is_masked) {
            visited_set_size++;
        }

        if (prev_visited) {
            continue;
        }

        if (is_masked) {
            vt.set(v1);
        }
        node_array[size++] = v1;

    }

    // add the first-hop neighbors to result/candidate heap, as well as sorting them to find the suitable two-hop neighbors
    nres += navix_batch_directed_compute_distance_filter_opt(
        node_array,
        size,
        sel,
        query_id,
        qdis,
        nbrs_to_explore,
        candidates,
        res,
        stats,
        vt);

    while (!nbrs_to_explore.empty()) {
        auto nbrs = nbrs_to_explore.top();
        nbrs_to_explore.pop();

        if (visited_set_size >= filter_nbrs_to_find) {
            break;
        }
        if (vt.get(nbrs.id)) {
            continue;
        }
        vt.set(nbrs.id);

        size_t second_begin, second_end;
        neighbor_range(nbrs.id, 0, &second_begin, &second_end);
        stats.nhops += 1;

        for (size_t j = second_begin; j < second_end; ++j) {
            int v2 = neighbors[j];
            if (v2 < 0) {
                second_end = j;
                break;
            }
            prefetch_L2(vt.visited.data() + v2);
        }

        for (size_t j = second_begin; j < second_end; ++j) {
            auto v2 = neighbors[j];
            nfilter++;
            bool prev_visited = vt.get_either(v2);
            bool filter_mask = get_and_set_member(sel, vt, v2, nfilter, query_id);
            if (filter_mask) {
                visited_set_size++;
            }
            if (prev_visited) {
                continue;
            }
            if (filter_mask) {
                if (size < node_array.size()) node_array[size++] = v2;
                vt.set(v2);
            }
        }
    }
    stats.nfilter += nfilter;
    return nres;
}

void NaviX::navix_blind(
    size_t begin,
    size_t end,
    VisitedTable &vt,
    const char *filter_id_map,
    int filter_nbrs_to_find,
    node_array_t &node_array,
    int &size,
    NaviXStats &stats) const {
    std::queue<storage_idx_t> nbrs_to_explore;
    // std::unordered_set<idx_t> visitedSet;

    // First Hop Neighbors
    int visited_set_size = 0;
    for (size_t j = begin; j < end; ++j) {
        auto v1 = neighbors[j];
        auto is_masked = filter_id_map[v1];
        if (is_masked) {
            visited_set_size++;
        }

        if (vt.get(v1)) {
            continue;
        }

        if (is_masked) {
            vt.set(v1);
            node_array[size++] = v1;
        }
        nbrs_to_explore.push(v1);
    }

    while (!nbrs_to_explore.empty()) {
        auto nbr = nbrs_to_explore.front();
        nbrs_to_explore.pop();

        if (visited_set_size >= filter_nbrs_to_find) {
            break;
        }
        if (vt.get(nbr)) {
            continue;
        }
        vt.set(nbr);

        size_t second_begin, second_end;
        neighbor_range(nbr, 0, &second_begin, &second_end);
        stats.nhops += 1;

        for (size_t j = second_begin; j < second_end; ++j) {
            int v2 = neighbors[j];
            if (v2 < 0) {
                second_end = j;
                break;
            }
            prefetch_L2(vt.visited.data() + v2);
            prefetch_L2(filter_id_map + v2);
        }

        for (size_t j = second_begin; j < second_end; ++j) {
            auto v2 = neighbors[j];
            auto is_masked = filter_id_map[v2];
            if (is_masked) {
                visited_set_size++;
            }
            if (vt.get(v2)) {
                continue;
            }
            if (is_masked) {
                node_array[size++] = v2;
                vt.set(v2);
            }
        }
    }
}

void NaviX::navix_full_two_hop(
    size_t begin,
    size_t end,
    VisitedTable &vt,
    const IDSelector *sel,
    int query_id,
    node_array_t &node_array,
    int &size,
    NaviXStats &stats) const {

    int nfilter = 0;
    for (size_t i = begin; i < end; i++) {
        auto v1 = neighbors[i];

        if(!vt.get(v1)){
            nfilter++;
            if (sel->is_member(v1, query_id)) {
                vt.set(v1);
                if (size < node_array.size()) node_array[size++] = v1;
            }
        }

        size_t second_begin, second_end;
        neighbor_range(v1, 0, &second_begin, &second_end);
        stats.nhops += 1;

        for (size_t j = second_begin; j < second_end; ++j) {
            int v2 = neighbors[j];
            if (v2 < 0) {
                second_end = j;
                break;
            }
            prefetch_L2(vt.visited.data() + v2);
        }

        for (size_t j = second_begin; j < second_end; ++j) {
            auto v2 = neighbors[j];

            if (!vt.get(v2)){
                nfilter++;
                if (sel->is_member(v2, query_id)) {
                    vt.set(v2);
                    if (size < node_array.size()) node_array[size++] = v2;
                }
            }

        }
    }
    stats.nfilter += nfilter;
}

void NaviX::navix_full_two_hop_filter_opt(
    size_t begin,
    size_t end,
    VisitedTable &vt,
    const IDSelector *sel,
    int query_id,
    node_array_t &node_array,
    int &size,
    NaviXStats &stats) const {

    int nfilter = 0;
    for (size_t i = begin; i < end; i++) {
        auto v1 = neighbors[i];

        if(!vt.get(v1)){
            nfilter++;
            if (get_and_set_member(sel, vt, v1, nfilter, query_id)) {
                vt.set(v1);
                if (size < node_array.size()) node_array[size++] = v1;
            }
        }

        size_t second_begin, second_end;
        neighbor_range(v1, 0, &second_begin, &second_end);
        stats.nhops += 1;

        for (size_t j = second_begin; j < second_end; ++j) {
            int v2 = neighbors[j];
            if (v2 < 0) {
                second_end = j;
                break;
            }
            prefetch_L2(vt.visited.data() + v2);
        }

        for (size_t j = second_begin; j < second_end; ++j) {
            auto v2 = neighbors[j];

            if (!vt.get(v2)){
                nfilter++;
                if (get_and_set_member(sel, vt, v2, nfilter, query_id)) {
                    vt.set(v2);
                    if (size < node_array.size()) node_array[size++] = v2;
                }
            }

        }
    }
    stats.nfilter += nfilter;
}

int NaviX::navix_add_filtered_nodes_to_candidates(
    DistanceComputer &qdis,
    MinimaxHeap &candidates,
    ResultHandler<C> &res,
    VisitedTable &vt,
    // const char* filter_id_map,//lbj+
    const IDSelector *sel,
    int query_id,
    int num_of_nodes,
    NaviXStats &stats) const {

    auto threshold = res.threshold;
    auto nres = 0;

    int ndist = 0;
    int nfilter = 0;

    // Add some random filtered nodes to candidates and results pq
    auto ntotal = levels.size();
    int count = 0;
    for (storage_idx_t p_id = 0; p_id < ntotal; p_id++) {
        if (vt.get(p_id)) {
            continue;
        }
        nfilter++;
        if (sel->is_member(p_id, query_id)){
            ndist++;
            float dist = qdis(p_id);

            if (dist < threshold) {
                if (res.add_result(dist, p_id)) {
                    threshold = res.threshold;
                    nres++;
                }
            }
            candidates.push(p_id, dist);

            vt.set(p_id);
            count++;
        }
        if (count >= num_of_nodes) {
            break;
        }
    }

    stats.nfilter += nfilter;
    stats.ndis += ndist;

    return nres;
}

int NaviX::navix_add_filtered_nodes_to_candidates_filter_opt(
    DistanceComputer &qdis,
    MinimaxHeap &candidates,
    ResultHandler<C> &res,
    VisitedTable &vt,
    // const char* filter_id_map,//lbj+
    const IDSelector *sel,
    int query_id,
    int num_of_nodes,
    NaviXStats &stats) const {

    auto threshold = res.threshold;
    auto nres = 0;

    int ndist = 0;
    int nfilter = 0;

    // Add some random filtered nodes to candidates and results pq
    auto ntotal = levels.size();
    int count = 0;
    for (storage_idx_t p_id = 0; p_id < ntotal; p_id++) {

        if (vt.get(p_id)) {
            continue;
        }

        if (get_and_set_member(sel, vt, p_id, nfilter, query_id)){
            ndist++;
            float dist = qdis(p_id);

            if (dist < threshold) {
                if (res.add_result(dist, p_id)) {
                    threshold = res.threshold;
                    nres++;
                }
            }
            candidates.push(p_id, dist);

            count++;
        }
        if (count >= num_of_nodes) {
            break;
        }
    }

    stats.nfilter += nfilter;
    stats.ndis += ndist;

    return nres;
}

NaviXStats NaviX::navix_hybrid_search(
    DistanceComputer &qdis,
    ResultHandler<C> &res,
    VisitedTable &vt,
    const IDSelector* sel,
    int query_id,
    const SearchParametersNaviX *params) const {
    NaviXStats stats;

    FAISS_ASSERT(sel);

    if (entry_point == -1) {
        return stats;
    }

    int run_efSearch = this->efSearch;
    int nfilter = 0;

    if (params) {
        if (const SearchParametersNaviX *navix_params =
                dynamic_cast<const SearchParametersNaviX *>(params)) {
            run_efSearch = navix_params->efSearch;
        }
    }

    // First search on the upper layer!!
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for (int level = max_level; level >= 1; level--) {
        NaviXStats local_stats = greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        stats.combine(local_stats);
    }

    auto k = extract_k_from_ResultHandler(res);
    int ef = std::max(run_efSearch, k);

    // Add the nearest node to the candidates and results pq
    MinimaxHeap candidates(ef);
    candidates.push(nearest, d_nearest);

    nfilter++;
    if (sel->is_member(nearest, query_id)) {
        res.add_result(d_nearest, nearest);
        vt.set(nearest);
    }

    // add random nodes to behave as the entry points
    navix_add_filtered_nodes_to_candidates(
        qdis, candidates, res, vt, sel, query_id, 10, stats);

    // Initialize the node array
    node_array_t node_array;
    int size = 0;

    while (candidates.size() > 0) {
        float cand_dist = 0;
        storage_idx_t candidate = candidates.pop_min(&cand_dist);

        // tricky stopping condition: there are more that ef
        // distances that are processed already that are smaller
        // than cand_dist
        int n_dis_below = candidates.count_below(cand_dist);
        if (n_dis_below >= run_efSearch) {
            break;
        }

        size_t begin, end;
        neighbor_range(candidate, 0, &begin, &end);
        stats.nhops += 1;

        for (storage_idx_t j = begin; j < end; ++j) {
            storage_idx_t v1 = neighbors[j];
            if (v1 < 0) {
             end = j;
             break;
            }
            prefetch_L2(vt.visited.data() + v1);
        }

        // Calculate local selectivity
        double total_nbrs = end - begin;
        double filtered_nbrs = 0;

        for (size_t j = begin; j < end; ++j) {
            int v1 = neighbors[j];
            nfilter++;
            if (sel->is_member(v1, query_id)) {
                filtered_nbrs++;
            }
        }

        double local_selectivity = filtered_nbrs / total_nbrs;
        auto estimated_full_two_hop_distance_comp = (total_nbrs * filtered_nbrs + filtered_nbrs) * 0.4;
        auto estimated_directed_distance_comp = total_nbrs + (total_nbrs - filtered_nbrs);

        if (local_selectivity >= 0.5) {
            navix_one_hop(begin, end, vt, sel, query_id, node_array, size, stats);
        } else if (estimated_full_two_hop_distance_comp > estimated_directed_distance_comp) {
            // Directed Two Hop
            navix_directed(
                begin,
                end,
                qdis,
                candidates,
                res,
                vt,
                sel,
                query_id,
                total_nbrs,
                node_array,
                size,
                stats);

        } else {
            // Blind Two Hop
            navix_full_two_hop(
                begin,
                end,
                vt,
                sel,
                query_id,
                node_array,
                size,
                stats);
        }

        navix_batch_compute_distance(
            node_array,
            size,
            qdis,
            candidates,
            res,
            stats);
    }

    // add counting of nr_filter
    stats.nfilter += nfilter;

    return stats;
}

NaviXStats NaviX::navix_hybrid_search_filter_opt(
    DistanceComputer &qdis,
    ResultHandler<C> &res,
    VisitedTable &vt,
    const IDSelector* sel,
    int query_id,
    const SearchParametersNaviX *params) const {
    NaviXStats stats;

    FAISS_ASSERT(sel);

    if (entry_point == -1) {
        return stats;
    }

    int run_efSearch = this->efSearch;
    int nfilter = 0;

    if (params) {
        if (const SearchParametersNaviX *navix_params =
                dynamic_cast<const SearchParametersNaviX *>(params)) {
            run_efSearch = navix_params->efSearch;
        }
    }

    // First search on the upper layer!!
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for (int level = max_level; level >= 1; level--) {
        NaviXStats local_stats = greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        stats.combine(local_stats);
    }

    auto k = extract_k_from_ResultHandler(res);
    int ef = std::max(run_efSearch, k);

    // Add the nearest node to the candidates and results pq
    MinimaxHeap candidates(ef);
    candidates.push(nearest, d_nearest);

    if (get_and_set_member(sel, vt, nearest, nfilter, query_id)) {
        res.add_result(d_nearest, nearest);
    }

    // add random nodes to behave as the entry points
    navix_add_filtered_nodes_to_candidates_filter_opt(
        qdis, candidates, res, vt, sel, query_id, 10, stats);

    // Initialize the node array
    node_array_t node_array;
    int size = 0;

    while (candidates.size() > 0) {
        float cand_dist = 0;
        storage_idx_t candidate = candidates.pop_min(&cand_dist);

        // tricky stopping condition: there are more that ef
        // distances that are processed already that are smaller
        // than cand_dist
        int n_dis_below = candidates.count_below(cand_dist);
        if (n_dis_below >= run_efSearch) {
            break;
        }

        size_t begin, end;
        neighbor_range(candidate, 0, &begin, &end);
        stats.nhops += 1;

        for (storage_idx_t j = begin; j < end; ++j) {
            storage_idx_t v1 = neighbors[j];
            if (v1 < 0) {
             end = j;
             break;
            }
            prefetch_L2(vt.visited.data() + v1);
        }

        // Calculate local selectivity
        double total_nbrs = end - begin;
        double filtered_nbrs = 0;

        for (size_t j = begin; j < end; ++j) {
            int v1 = neighbors[j];
            // we must retain this filter, or it will cause all these neighbors not to be tarversed
            nfilter++;
            if (sel->is_member(v1, query_id)) {
                filtered_nbrs++;
            }
        }

        double local_selectivity = filtered_nbrs / total_nbrs;
        auto estimated_full_two_hop_distance_comp = (total_nbrs * filtered_nbrs + filtered_nbrs) * 0.4;
        auto estimated_directed_distance_comp = total_nbrs + (total_nbrs - filtered_nbrs);

        if (local_selectivity >= 0.5) {
            navix_one_hop_filter_opt(begin, end, vt, sel, query_id, node_array, size, stats);
        } else if (estimated_full_two_hop_distance_comp > estimated_directed_distance_comp) {
            // Directed Two Hop
            navix_directed_filter_opt(
                begin,
                end,
                qdis,
                candidates,
                res,
                vt,
                sel,
                query_id,
                total_nbrs,
                node_array,
                size,
                stats);

        } else {
            // Blind Two Hop
            navix_full_two_hop_filter_opt(
                begin,
                end,
                vt,
                sel,
                query_id,
                node_array,
                size,
                stats);
        }

        navix_batch_compute_distance(
            node_array,
            size,
            qdis,
            candidates,
            res,
            stats);
    }

    // add counting of nr_filter
    stats.nfilter += nfilter;

    return stats;
}

void NaviX::search_level_0(
        DistanceComputer& qdis,
        ResultHandler<C>& res,
        idx_t nprobe,
        const storage_idx_t* nearest_i,
        const float* nearest_d,
        int search_type,
        NaviXStats& search_stats,
        VisitedTable& vt,
        const SearchParameters* params) const {
    const NaviX& navix = *this;

    auto efSearch = navix.efSearch;
    if (params) {
        if (const SearchParametersNaviX* navix_params =
                    dynamic_cast<const SearchParametersNaviX*>(params)) {
            efSearch = navix_params->efSearch;
        }
    }

    int k = extract_k_from_ResultHandler(res);

    if (search_type == 1) {
        int nres = 0;

        for (int j = 0; j < nprobe; j++) {
            storage_idx_t cj = nearest_i[j];

            if (cj < 0)
                break;

            if (vt.get(cj))
                continue;

            int candidates_size = std::max(efSearch, k);
            MinimaxHeap candidates(candidates_size);

            candidates.push(cj, nearest_d[j]);

            nres = search_from_candidates(
                    navix,
                    qdis,
                    res,
                    candidates,
                    vt,
                    search_stats,
                    0,
                    nres,
                    params);
            nres = std::min(nres, candidates_size);
        }
    } else if (search_type == 2) {
        int candidates_size = std::max(efSearch, int(k));
        candidates_size = std::max(candidates_size, int(nprobe));

        MinimaxHeap candidates(candidates_size);
        for (int j = 0; j < nprobe; j++) {
            storage_idx_t cj = nearest_i[j];

            if (cj < 0)
                break;
            candidates.push(cj, nearest_d[j]);
        }

        search_from_candidates(
                navix, qdis, res, candidates, vt, search_stats, 0, 0, params);
    }
}

void NaviX::permute_entries(const idx_t* map) {
    // remap levels
    storage_idx_t ntotal = levels.size();
    std::vector<storage_idx_t> imap(ntotal); // inverse mapping
    // map: new index -> old index
    // imap: old index -> new index
    for (int i = 0; i < ntotal; i++) {
        assert(map[i] >= 0 && map[i] < ntotal);
        imap[map[i]] = i;
    }
    if (entry_point != -1) {
        entry_point = imap[entry_point];
    }
    std::vector<int> new_levels(ntotal);
    std::vector<size_t> new_offsets(ntotal + 1);
    std::vector<storage_idx_t> new_neighbors(neighbors.size());
    size_t no = 0;
    for (int i = 0; i < ntotal; i++) {
        storage_idx_t o = map[i]; // corresponding "old" index
        new_levels[i] = levels[o];
        for (size_t j = offsets[o]; j < offsets[o + 1]; j++) {
            storage_idx_t neigh = neighbors[j];
            new_neighbors[no++] = neigh >= 0 ? imap[neigh] : neigh;
        }
        new_offsets[i + 1] = no;
    }
    assert(new_offsets[ntotal] == offsets[ntotal]);
    // swap everyone
    std::swap(levels, new_levels);
    std::swap(offsets, new_offsets);
    neighbors = std::move(new_neighbors);
}
//lbj-


/**************************************************************
 * MinimaxHeap
 **************************************************************/

void NaviX::MinimaxHeap::push(storage_idx_t i, float v) {
    if (k == n) {
        if (v >= dis[0])
            return;
        faiss::heap_pop<HC>(k--, dis.data(), ids.data());
        --nvalid;
    }
    faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
    ++nvalid;
}

float NaviX::MinimaxHeap::max() const {
    return dis[0];
}

int NaviX::MinimaxHeap::size() const {
    return nvalid;
}

void NaviX::MinimaxHeap::clear() {
    nvalid = k = 0;
}

#ifdef __AVX2__

int NaviX::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);
    static_assert(
            std::is_same<storage_idx_t, int32_t>::value,
            "This code expects storage_idx_t to be int32_t");

    int32_t min_idx = -1;
    float min_dis = std::numeric_limits<float>::infinity();

    size_t iii = 0;

    __m256i min_indices = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, -1);
    __m256 min_distances =
            _mm256_set1_ps(std::numeric_limits<float>::infinity());
    __m256i current_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i offset = _mm256_set1_epi32(8);

    // The baseline version is available in non-AVX2 branch.

    // The following loop tracks the rightmost index with the min distance.
    // -1 index values are ignored.
    const int k8 = (k / 8) * 8;
    for (; iii < k8; iii += 8) {
        __m256i indices =
                _mm256_loadu_si256((const __m256i*)(ids.data() + iii));
        __m256 distances = _mm256_loadu_ps(dis.data() + iii);

        // This mask filters out -1 values among indices.
        __m256i m1mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), indices);

        __m256i dmask = _mm256_castps_si256(
                _mm256_cmp_ps(min_distances, distances, _CMP_LT_OS));
        __m256 finalmask = _mm256_castsi256_ps(_mm256_or_si256(m1mask, dmask));

        const __m256i min_indices_new = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(current_indices),
                _mm256_castsi256_ps(min_indices),
                finalmask));

        const __m256 min_distances_new =
                _mm256_blendv_ps(distances, min_distances, finalmask);

        min_indices = min_indices_new;
        min_distances = min_distances_new;

        current_indices = _mm256_add_epi32(current_indices, offset);
    }

    // Vectorizing is doable, but is not practical
    int32_t vidx8[8];
    float vdis8[8];
    _mm256_storeu_ps(vdis8, min_distances);
    _mm256_storeu_si256((__m256i*)vidx8, min_indices);

    for (size_t j = 0; j < 8; j++) {
        if (min_dis > vdis8[j] || (min_dis == vdis8[j] && min_idx < vidx8[j])) {
            min_idx = vidx8[j];
            min_dis = vdis8[j];
        }
    }

    // process last values. Vectorizing is doable, but is not practical
    for (; iii < k; iii++) {
        if (ids[iii] != -1 && dis[iii] <= min_dis) {
            min_dis = dis[iii];
            min_idx = iii;
        }
    }

    if (min_idx == -1) {
        return -1;
    }

    if (vmin_out) {
        *vmin_out = min_dis;
    }
    int ret = ids[min_idx];
    ids[min_idx] = -1;
    --nvalid;
    return ret;
}

#else

// baseline non-vectorized version
int NaviX::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);
    // returns min. This is an O(n) operation
    int i = k - 1;
    while (i >= 0) {
        if (ids[i] != -1) {
            break;
        }
        i--;
    }
    if (i == -1) {
        return -1;
    }
    int imin = i;
    float vmin = dis[i];
    i--;
    while (i >= 0) {
        if (ids[i] != -1 && dis[i] < vmin) {
            vmin = dis[i];
            imin = i;
        }
        i--;
    }
    if (vmin_out) {
        *vmin_out = vmin;
    }
    int ret = ids[imin];
    ids[imin] = -1;
    --nvalid;

    return ret;
}
#endif

int NaviX::MinimaxHeap::count_below(float thresh) {
    int n_below = 0;
    for (int i = 0; i < k; i++) {
        if (dis[i] < thresh) {
            n_below++;
        }
    }

    return n_below;
}

} // namespace faiss
