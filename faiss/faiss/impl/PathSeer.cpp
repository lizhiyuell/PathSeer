/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/PathSeer.h>

#include <string>

#ifdef __AVX2__
#include <immintrin.h>

#include <limits>
#include <type_traits>
#endif

// added
#include <sys/time.h>
#include <stdio.h>
#include <iostream>
#include <math.h>  
#include <unordered_map>
#include <iostream>
#include <fstream>

namespace faiss {


/**************************************************************
 * PathSeer structure implementation
 **************************************************************/

int PathSeer::nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no + 1] -
            cum_nneighbor_per_level[layer_no];
}

void PathSeer::set_nb_neighbors(int level_no, int n) {
    FAISS_THROW_IF_NOT(levels.size() == 0);
    int cur_n = nb_neighbors(level_no);
    for (int i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
        cum_nneighbor_per_level[i] += n - cur_n;
    }
}

int PathSeer::cum_nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no];
}

void PathSeer::neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end) const {
    size_t o = offsets[no];
    *begin = o + cum_nb_neighbors(layer_no);
    *end = o + cum_nb_neighbors(layer_no + 1);
}

int PathSeer::get_neighbor_dist_boundary(idx_t no, int layer_no) const{
    return neighbor_boundaries[offset_neighbor[no] + layer_no];
}

void PathSeer::set_neighbor_dist_boundary(idx_t no, int layer_no, int new_boundary){
    neighbor_boundaries[offset_neighbor[no] + layer_no] = new_boundary;
}

PathSeer::PathSeer(int M, int M_exp) : rng(12345) {
    set_default_probas(M, 1.0 / log(M), M_exp);
    max_level = -1;
    entry_point = -1;
    efSearch = 16;
    efConstruction = 100;
    upper_beam = 1;
    this->M = M;
    this->M_exp = M_exp;
    offsets.push_back(0);
    offset_neighbor.push_back(0);
    this->profilier = new static_profile_params;
}

PathSeer::~PathSeer(){
    // if(this->profilier){
    //     delete this->profilier;
    //     this->profilier = nullptr;
    // }
}

int PathSeer::random_level() {
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

void PathSeer::set_default_probas(int M, float levelMult, int M_exp) {
    int nn = 0;
    cum_nneighbor_per_level.push_back(0);
    if (M_exp < M) {
        printf("Error: M_exp %d is smaller than M %d!\n", M_exp, M);
        FAISS_THROW_MSG("Parameter error.");
    }
    for (int level = 0;; level++) {
        float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
        if (proba < 1e-9)
            break;
        assign_probas.push_back(proba);
        nn += level == 0 ? M_exp * 2 : M_exp;
        cum_nneighbor_per_level.push_back(nn);
    }
}

void PathSeer::clear_neighbor_tables(int level) {
    for (int i = 0; i < levels.size(); i++) {
        size_t begin, end;
        neighbor_range(i, level, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            neighbors[j] = PathSeer::NeighNode(-1);
        }
    }
}

void PathSeer::reset() {
    max_level = -1;
    entry_point = -1;
    offsets.clear();
    offsets.push_back(0);
    levels.clear();
    neighbors.clear();
    neighbor_boundaries.clear();
    offset_neighbor.clear();
    offset_neighbor.push_back(0);
}


void PathSeer::fill_with_random_links(size_t n) {
    throw FaissException("UNIMPLEMENTED");
}

// Modified from original HNSW
// n is the number of vectors that will be added (this gets called in IndexHSNW.hnsw_add_vertices)
int PathSeer::prepare_level_tab(size_t n, bool preset_levels) {
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
        if (pt_level > max_level){
            max_level = pt_level;
        }

        offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));

        neighbors.resize(offsets.back(), NeighNode(-1));

        offset_neighbor.push_back(offset_neighbor.back() + pt_level + 1);
        neighbor_boundaries.resize(offset_neighbor.back(), 0);

    }

    return max_level;
}

void PathSeer::HNSW_shrink_neighbor_list(
    DistanceComputer& qdis,
    std::priority_queue<NodeDistFarther>& input,
    std::vector<NodeDistFarther>& output,
    int max_size,
    std::vector<NodeDistFarther>& outsiders,
    bool keep_max_size_level0){

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
                    while(input.size() > 0){
                        outsiders.push_back(input.top());
                        input.pop();
                    }
                    return;
                }
            } else {
                outsiders.push_back(v1);
            }
        }
        // size_t idx = 0;
        // while (keep_max_size_level0 && (output.size() < max_size) &&
        //     (idx < outsiders.size())) {
        //     output.push_back(outsiders[idx++]);
        // }
    }

// this funciton does RNG-based pruning on input, but add the pruned elements into the other. Note that the total size of input and other has been restricted, so there is no size-impose on the other.
void PathSeer::PathSeer_shrink_neighbor_list(
    DistanceComputer& qdis,
    std::priority_queue<NodeDistFarther>& input,
    std::priority_queue<NodeDistCloser>& other,
    std::vector<NodeDistFarther>& output,
    int max_size){

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
                    // insert the remaining ones to the other vecs
                    while(!input.empty()){
                        other.push(NodeDistCloser(input.top().d, input.top().id));
                        input.pop();
                    }
                    return;
                }
            }
            // #if PATHSEER_CONSTRUCTION_METHOD!=1
            else{
                other.push(NodeDistCloser(v1.d, v1.id));
            }
            // #endif
        }
    }


namespace {

using storage_idx_t = PathSeer::storage_idx_t;
using NodeDistCloser = PathSeer::NodeDistCloser;
using NodeDistFarther = PathSeer::NodeDistFarther;

/**************************************************************
 * Addition subroutines
 **************************************************************/

// the retured value is the droped vectors, starting from nearest to fathest
void hnsw_shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& resultSet1,
        int max_size,
        std::vector<NodeDistFarther>& outsiders,
        bool keep_max_size_level0 = false) {
    if (resultSet1.size() < max_size) {
        return;
    }
    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    PathSeer::HNSW_shrink_neighbor_list(
            qdis, resultSet, returnlist, max_size, outsiders, keep_max_size_level0);

    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }

}

void pathseer_shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& resultSet1,
        std::priority_queue<NodeDistCloser>& resultOther,
        int max_size,
        bool keep_max_size_level0 = false) {
    if (resultSet1.size() < max_size) {
        return;
    }
    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    PathSeer::PathSeer_shrink_neighbor_list(
            qdis, resultSet, resultOther, returnlist, max_size);

    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }
}

// moving the extending zone by x, in which x>0 means moving to the right
void move_the_extending_zone(PathSeer& pathseer, size_t begin, size_t end, int dist_boundary, int x){

    if(!x)
        return;

    // find the true_end
    long long int true_begin = (long long int) (begin + dist_boundary);
    long long int true_end = end;
    while(true_end > begin + dist_boundary){
        if(pathseer.neighbors[true_end-1]!=-1)
            break;
        true_end--;
    }

    // if the extending zone is empty, just quit
    if(true_end==true_begin)
        return;

    FAISS_ASSERT(true_end > true_begin);

    if(x > 0){
        FAISS_ASSERT(dist_boundary + x <= end);
        
        // maybe we should drop some vectors
        size_t nr_moved = true_end - true_begin;
        if(nr_moved + x > end - true_begin)
            nr_moved = end - true_begin - x;
        
        std::memmove(pathseer.neighbors.data() + true_begin + x, pathseer.neighbors.data() + true_begin, sizeof(faiss::storage_idx_t) * nr_moved);

    }
    else if(x < 0){
        FAISS_ASSERT(x <= dist_boundary);

        // just move it
        std::memmove(pathseer.neighbors.data() + true_begin + x, pathseer.neighbors.data() + true_begin, sizeof(faiss::storage_idx_t) * (true_end - true_begin));

        // set the remaining neighbors to -1
        for(size_t i=true_end + x; i<true_end; i++)
            pathseer.neighbors[i] = -1;

    }

}


// ---------- multiple variant of add_links in PathSeer ----------
// add links in the dist-oriented area
void add_link_dist_oriented(
        PathSeer& pathseer,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level,
        bool keep_max_size_level0 = false) {

    // maximum position of dist-oriented zone
    int dist_boundary = level==0 ? pathseer.M*2 : pathseer.M;

    size_t begin, end;
    pathseer.neighbor_range(src, level, &begin, &end);

    // 1. there is still position in the dist-oriented zone.
    int prev_boundary = pathseer.get_neighbor_dist_boundary(src, level);
    if(prev_boundary < dist_boundary){
        // put the point into dist-oriented zone
        int saved_point = pathseer.neighbors[begin + prev_boundary];
        pathseer.neighbors[begin + prev_boundary] = dest;

        prev_boundary++;

        // std::atomic_thread_fence(std::memory_order_seq_cst);

        pathseer.set_neighbor_dist_boundary(src, level, prev_boundary);
        // printf("[Change 1] layer=%d, uppoer=%d, point %d, boundary %d\n", level, dist_boundary, src, pathseer.neighbor_boundaries[src]);
        // extend point to the expanding zone. Perhaps the last point will exceed the expanding zone, just drop it.
        size_t pos = begin + prev_boundary;
        while(saved_point!=-1 && pos < end){
            int tmp = pathseer.neighbors[pos];
            pathseer.neighbors[pos] = saved_point;
            saved_point = tmp;
            pos++;
        }
    }
    // 2. there is no room in the dist_boundary zone, prune one vector
    else{
        assert(prev_boundary == dist_boundary);

        // copy to resultSet...
        std::priority_queue<NodeDistCloser> resultSet;
        resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
        for (size_t i = begin; i < begin + dist_boundary; i++) {
            storage_idx_t neigh = pathseer.neighbors[i];
            resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
        }

        // shrink and get the residual
        std::vector<NodeDistFarther> residual;
        ::faiss::hnsw_shrink_neighbor_list(qdis, resultSet, dist_boundary, residual, keep_max_size_level0);

        // ...and back
        size_t i = begin;
        while (resultSet.size()) {
            pathseer.neighbors[i++] = resultSet.top().id;
            resultSet.pop();
        }

        // std::atomic_thread_fence(std::memory_order_seq_cst);

        // they may have shrunk more than just by 1 element, put this new boundary as the dist_boundary
        pathseer.set_neighbor_dist_boundary(src, level, i-begin);
        // pathseer.neighbor_boundaries[src] = i - begin;
        // printf("[Change 2] layer=%d, uppoer=%d, point %d, boundary %d\n", level, dist_boundary, src, pathseer.neighbor_boundaries[src]);

        // put the remaining vectors into the expanding zone. Note that this will only add one additional vector to the following zone
        assert(residual.size()+(i-begin)==dist_boundary+1);

        int j = 0;
        while(i < begin + dist_boundary){
            pathseer.neighbors[i] = residual[j].id;
            i++;
            j++;
        }

        assert(j == residual.size()-1);

        int saved_point = residual[j].id;
        while(saved_point!=-1 && i < end){
            int tmp = pathseer.neighbors[i];
            pathseer.neighbors[i] = saved_point;
            saved_point = tmp;
            i++;
        }
    }
}


// a better version of add_link_dist_oriented, that keeps the order of the extending area.
void add_link_dist_oriented_opt(
        PathSeer& pathseer,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level,
        bool keep_max_size_level0 = false) {

    // maximum position of dist-oriented zone
    int dist_boundary = level==0 ? pathseer.M*2 : pathseer.M;

    size_t begin, end;
    pathseer.neighbor_range(src, level, &begin, &end);

    // 1. there is still position in the dist-oriented zone.
    int prev_boundary = pathseer.get_neighbor_dist_boundary(src, level);
    if(prev_boundary < dist_boundary){

        // move the extending zone by 1
        move_the_extending_zone(pathseer, begin, end, prev_boundary, 1);

        // put the point into dist-oriented zone
        int saved_point = pathseer.neighbors[begin + prev_boundary];
        pathseer.neighbors[begin + prev_boundary] = dest;

        pathseer.set_neighbor_dist_boundary(src, level, prev_boundary + 1);

    }
    // 2. there is no room in the dist_boundary zone, prune one vector
    else{

        assert(prev_boundary == dist_boundary);

        // copy to resultSet...
        std::priority_queue<NodeDistCloser> resultSet;
        resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
        for (size_t i = begin; i < begin + prev_boundary; i++) {
            storage_idx_t neigh = pathseer.neighbors[i];
            resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
        }

        // shrink and get the residual
        std::vector<NodeDistFarther> residual;
        ::faiss::hnsw_shrink_neighbor_list(qdis, resultSet, dist_boundary, residual, keep_max_size_level0);

        int new_boundary = resultSet.size();
        move_the_extending_zone(pathseer, begin, end, prev_boundary, new_boundary - prev_boundary);

        for(long long int i=(long long int)(begin + new_boundary-1); i>=(long long int)begin; i--){
            storage_idx_t dest_id = resultSet.top().id;
            resultSet.pop();
            pathseer.neighbors[i] = dest_id;
        }
        FAISS_ASSERT(resultSet.empty());

        pathseer.set_neighbor_dist_boundary(src, level, new_boundary);

    }
}

void add_link_dist_oriented_batched(
        PathSeer& pathseer,
        DistanceComputer& qdis,
        storage_idx_t src,
        std::priority_queue<NodeDistCloser>& dest_heap,
        std::vector<storage_idx_t>& neighbors,
        std::priority_queue<NodeDistCloser>& other_link_targets,
        int level) {

    // maximum position of dist-oriented zone
    int dist_boundary = level==0 ? pathseer.M*2 : pathseer.M;

    size_t begin, end;
    pathseer.neighbor_range(src, level, &begin, &end);
    int prev_boundary = pathseer.get_neighbor_dist_boundary(src, level);

    // if the dist_oriented zone is not full after adding new vectors, we don't prune them
    if(prev_boundary + dest_heap.size() < dist_boundary){
        int new_boundary = prev_boundary + dest_heap.size();
        // move the extending_zone back
        move_the_extending_zone(pathseer, begin, end, prev_boundary, new_boundary - prev_boundary);
        // just put the points into the dist-oriented zone
        for(int i = new_boundary-1; i>=prev_boundary; i--){
            storage_idx_t dest_id = dest_heap.top().id;
            pathseer.neighbors[begin + i] = dest_id;
            dest_heap.pop();
            neighbors.push_back(dest_id);
        }
        FAISS_ASSERT(dest_heap.empty());
        pathseer.set_neighbor_dist_boundary(src, level, new_boundary);
    }
    else{
        // we should get all previous points and put them into the heap
        for(size_t i=begin; i<begin+prev_boundary; i++){
            auto v = pathseer.neighbors[i];
            dest_heap.push(NodeDistCloser(qdis.symmetric_dis(src, v), v));
        }

        // shrink the neighbor list
        std::vector<NodeDistFarther> residual;
        ::faiss::hnsw_shrink_neighbor_list(qdis, dest_heap, dist_boundary, residual);

        // we put the residual vectors into the other link targets for futher processing
        for(auto it : residual){
            other_link_targets.push(NodeDistCloser(it.d, it.id));
        }

        int new_boundary = dest_heap.size();
        move_the_extending_zone(pathseer, begin, end, prev_boundary, new_boundary - prev_boundary);

        for(long long int i=(long long int)(begin + new_boundary-1); i>=(long long int)begin; i--){
            storage_idx_t dest_id = dest_heap.top().id;
            dest_heap.pop();
            pathseer.neighbors[i] = dest_id;
            neighbors.push_back(dest_id);
        }
        FAISS_ASSERT(dest_heap.empty());

        pathseer.set_neighbor_dist_boundary(src, level, new_boundary);

    }

}

// add links in the extending area
void add_link_extending(
        PathSeer& pathseer,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level) {

    size_t begin, end;
    pathseer.neighbor_range(src, level, &begin, &end);

    int dist_boundary = pathseer.use_only_expanding_idx ? 0 : pathseer.get_neighbor_dist_boundary(src, level);

    // start from the dist_boundary position
    begin += dist_boundary;
    if (pathseer.neighbors[end - 1] == -1) {
        // there is enough room, find a slot to add it
        long long int i = end;
        while (i > (long long int)begin) {
            if (pathseer.neighbors[i - 1] != -1)
                break;
            i--;
        }
        pathseer.neighbors[i] = dest;
        return;
    }

    // otherwise we let them fight out which to keep

    // copy to resultSet...
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) {
        auto neigh = pathseer.neighbors[i];
        FAISS_ASSERT(neigh!=-1);
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    FAISS_ASSERT(resultSet.size()==end-begin+1);
    while(resultSet.size() > end - begin)
        resultSet.pop();

    // put the result into the neighbors
    long long int pos = end - 1;
    while(!resultSet.empty() && pos >= begin){
        pathseer.neighbors[pos--] = resultSet.top().id;
        resultSet.pop();
    }
    FAISS_ASSERT(pos==begin-1);

}

// add links in the extending area in a batch manner
void add_link_extending_batched(
        PathSeer& pathseer,
        DistanceComputer& qdis,
        storage_idx_t src,
        std::priority_queue<NodeDistCloser>& dest_heap,
        std::vector<storage_idx_t>& neighbors,
        int level) {

    size_t begin, end;
    pathseer.neighbor_range(src, level, &begin, &end);

    // here we use 0 dist_boundary because all neighbors are in the expanding zone
    int dist_boundary = pathseer.use_only_expanding_idx ? 0 : pathseer.get_neighbor_dist_boundary(src, level);

    // 1. use a priority_queue to maintain both the new vectors and previous vectors
    std::priority_queue<NodeDistCloser> result_heap;
    while(!dest_heap.empty()){
        result_heap.push(dest_heap.top());
        neighbors.push_back(dest_heap.top().id);
        dest_heap.pop();
    }

    // here we just use the dest_heap as our ordering heap
    for (size_t i = begin + dist_boundary; i < end; i++) {
        auto neigh = pathseer.neighbors[i];
        if(neigh==-1)
            break;
        result_heap.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    // 2. removing the excessive ones
    while(result_heap.size() > end - begin - dist_boundary)
        result_heap.pop();

    // 3. put all the vectors into the expanding zone
    long long int last_pos_exclusive = begin + dist_boundary + result_heap.size();
    for(long long int i = (long long int)(last_pos_exclusive - 1); i>=(long long int)(begin + dist_boundary); i--){
        pathseer.neighbors[i] = result_heap.top().id;
        result_heap.pop();
    }
    FAISS_ASSERT(result_heap.empty());

    // 4. marking the remaining vectors as -1
    while(last_pos_exclusive < end){
        pathseer.neighbors[last_pos_exclusive++] = PathSeer::NeighNode(-1);
    }

}

void add_link_extending_fast(
        PathSeer& pathseer,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level) {

    size_t begin, end;
    pathseer.neighbor_range(src, level, &begin, &end);

    int dist_boundary = pathseer.use_only_expanding_idx ? 0 : pathseer.get_neighbor_dist_boundary(src, level);

    // start from the dist_boundary position
    long long int true_begin = begin + dist_boundary;

    // there is no element in the expanding zone, just put the new element in it
    if(pathseer.neighbors[true_begin]==-1){
        pathseer.neighbors[true_begin] = dest;
        return;
    }

    // find the true end position, exclusive
    long long int true_end = end;
    if (pathseer.neighbors[true_end - 1] == -1) {
        // there is enough room, find a slot to add it
        while (true_end > true_begin) {
            if (pathseer.neighbors[true_end - 1] != -1)
                break;
            true_end--;
        }
    }

    FAISS_ASSERT(true_end > true_begin);

    // we use the binary search to find a position to insert
    float target_dist = qdis.symmetric_dis(src, dest);
    size_t left = true_begin, right = true_end;
    while(left < right){
        size_t mid = (left + right) >> 1;
        float _dist = qdis.symmetric_dis(src, pathseer.neighbors[mid]);

        if(_dist < target_dist){
            left = mid + 1;
        }
        else
            right = mid;
    }

    // here, the position "left" is the first position whose distance is larger than the target

    // maybe the "true_end" equals "end", we should shrink one element to drop the last element
    if(true_end==end)
        true_end--;

    // perhaps the newly added point is the largest one, in this case we just quit
    if(left > true_end)
        return;
    else if(left < true_end)
        std::memmove(pathseer.neighbors.data() + left + 1, pathseer.neighbors.data() + left, sizeof(faiss::storage_idx_t) * (true_end - left));

    // put the newly added element
    pathseer.neighbors[left] = dest;

}

void pathseer_search_neighbors_to_add_only_expanding(
        PathSeer& pathseer,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results_other_vecs,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt) {

    // top is nearest candidate
    std::priority_queue<NodeDistFarther> candidates;

    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results_other_vecs.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    // number of vectors need for linking
    int max_nr_neighbor = std::max<int>(pathseer.nb_neighbors(level), pathseer.efConstructionExp);

    while (!candidates.empty()) {
        // get nearest
        const NodeDistFarther& currEv = candidates.top();

        if (currEv.d > results_other_vecs.top().d) {
            break;
        }
    
        int currNode = currEv.id;
        candidates.pop();

        // loop over neighbors
        size_t begin, end;
        pathseer.neighbor_range(currNode, level, &begin, &end);

        for (size_t i = begin; i < end; i++) {
            auto nodeId = pathseer.neighbors[i];
            if(nodeId < 0)
                break;
            if (vt.get(nodeId))
                continue;
            vt.set(nodeId);

            float dis = qdis(nodeId);

            if(results_other_vecs.size() < max_nr_neighbor || results_other_vecs.top().d > dis) {
                results_other_vecs.emplace(dis, nodeId);
                candidates.emplace(dis, nodeId);
                if (results_other_vecs.size() > max_nr_neighbor) {
                    results_other_vecs.pop();
                }
            }
        }
    }

    vt.advance();
}

void pathseer_search_neighbors_to_add(
        PathSeer& pathseer,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results_dist_vecs,
        std::priority_queue<NodeDistCloser>& results_other_vecs,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt) {

    // top is nearest candidate
    std::priority_queue<NodeDistFarther> candidates;

    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results_dist_vecs.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    // number of vectors need for linking
    int max_nr_neighbor = std::max<int>(pathseer.nb_neighbors(level), pathseer.efConstructionExp);

    // step 1: simultanenously build the two arrays
    bool building_dist_oriented = true;
    while (!candidates.empty()) {
        // get nearest
        const NodeDistFarther& currEv = candidates.top();

        // if (results_other_vecs.size() >= max_nr_neighbor) {

        if ((building_dist_oriented && currEv.d > results_dist_vecs.top().d)) {
            building_dist_oriented = false;
        }
        else if(!building_dist_oriented && results_other_vecs.size() >= max_nr_neighbor)
            break;

        int currNode = currEv.id;
        candidates.pop();

        // loop over neighbors
        size_t begin, end;
        pathseer.neighbor_range(currNode, level, &begin, &end);
        end = begin + pathseer.get_neighbor_dist_boundary(currNode, level);

        for (size_t i = begin; i < end; i++) {
            auto nodeId = pathseer.neighbors[i];
            if(nodeId < 0)
                break;
            if (vt.get(nodeId))
                continue;
            vt.set(nodeId);

            float dis = qdis(nodeId);
            // NodeDistFarther evE1(dis, nodeId);

            if(building_dist_oriented){
                if(results_dist_vecs.size() < pathseer.efConstruction || results_dist_vecs.top().d > dis){
                    results_dist_vecs.emplace(dis, nodeId);
                    candidates.emplace(dis, nodeId);
                    if (results_dist_vecs.size() > pathseer.efConstruction) {
                        // when pop-out the element, push it into the other vecs
                        float _dist = results_dist_vecs.top().d;
                        float _id = results_dist_vecs.top().id;
                        results_other_vecs.emplace(NodeDistCloser(_dist, _id));
                        results_dist_vecs.pop();
                    }
                }
            }
            else{
                if(results_other_vecs.size() < max_nr_neighbor || results_other_vecs.top().d > dis) {
                    results_other_vecs.emplace(dis, nodeId);
                    candidates.emplace(dis, nodeId);
                    if (results_other_vecs.size() > max_nr_neighbor) {
                        results_other_vecs.pop();
                    }
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
/// used for construction (other version below will be used in search)
PathSeerStats greedy_update_nearest(
        const PathSeer& pathseer,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {

    PathSeerStats stats;
    int ndis = 0;
    int nhop = 0;

    for (;;) {
        storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        pathseer.neighbor_range(nearest, level, &begin, &end);

        end = begin + pathseer.get_neighbor_dist_boundary(nearest, level);

        for (size_t i = begin; i < end; i++) {
            auto v = pathseer.neighbors[i];
            if(v < 0)
                break;
            float dis = qdis(v);
            ndis++;
            if (dis < d_nearest) {
                nearest = v;
                d_nearest = dis;
            }
        }

        nhop++;

        if (nearest == prev_nearest) {
            stats.ndis = ndis;
            stats.ndis_first = ndis;
            stats.nhop = nhop;
            stats.nhop_first = nhop;
            return stats;
        }
    }
}

PathSeerStats adaptive_hybrid_greedy_update_nearest(
        const PathSeer& pathseer,
        DistanceComputer& qdis,
        IDSelector* sel,
        int pos,
        int level,
        storage_idx_t& nearest,
        float& d_nearest,
        int& M_exp_search,
        long long int& dist_comp_ns,
        long long int& filter_time_ns,
        bool search_with_overhead_bounded) {

    PathSeerStats stats;
    size_t ndis = 0, nhop = 0;
    int nr_filter = 0;

    if(level==1 && dist_comp_ns==-1 && search_with_overhead_bounded){

        size_t begin, end;
        // as manually let it search from the base layer for profiling, as the upper layer may lack neighbor
        pathseer.neighbor_range(nearest, 0, &begin, &end);

        int dist_boundary = pathseer.get_neighbor_dist_boundary(nearest, 0);

        int nr_profiling = 5;
        int max_position = std::min<int>(dist_boundary, nr_profiling);
        auto time_start = std::chrono::high_resolution_clock::now();
        for(size_t i=begin; i<begin+max_position; i++){
            #ifdef ENABLE_PREFETCH
            if(i + 1 < begin + max_position)
                qdis.prefetch_vector(pathseer.neighbors[i+1]);
            #endif
            volatile float _dist = qdis(pathseer.neighbors[i]);
        }
        auto time_end = std::chrono::high_resolution_clock::now();

        dist_comp_ns = (long long int)std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start).count() / max_position;

    }

    // profile the filter time
    if(level==1 && filter_time_ns==-1 && search_with_overhead_bounded){

        size_t begin, end;
        // as manually let it search from the base layer for profiling, as the upper layer may lack neighbor
        pathseer.neighbor_range(nearest, 0, &begin, &end);

        int dist_boundary = pathseer.get_neighbor_dist_boundary(nearest, 0);

        int nr_profiling = 5;
        int max_position = std::min<int>(dist_boundary, nr_profiling);
        auto time_start = std::chrono::high_resolution_clock::now();
        for(size_t i=begin; i<begin+max_position; i++){
            volatile bool is_target = sel->is_member(pathseer.neighbors[i], pos);
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        nr_filter += max_position;

        filter_time_ns = (long long int)std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start).count() / max_position;

    }

    if(level==1 && search_with_overhead_bounded)
        M_exp_search = pathseer.M_exp;

    FAISS_ASSERT(M_exp_search >= 0 && M_exp_search <= pathseer.M_exp);

    for (;;) {

        storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        pathseer.neighbor_range(nearest, level, &begin, &end);

        end = begin + pathseer.get_neighbor_dist_boundary(nearest, level);

        for (size_t i = begin; i < end; i++) {
            storage_idx_t v = pathseer.neighbors[i];
            if (v < 0){
                break;
            }
            float dis = qdis(v);
            #ifdef OUTPUT_SEARCH_PATH
            printf("%d\t%f\t%d\n", v, dis, sel->is_member(v, pos)?0:1);
            #endif
            ndis++;
            if (dis < d_nearest) {
                nearest = v;
                d_nearest = dis;
            }
        }
        nhop++;

        if (nearest == prev_nearest) {
            stats.ndis = ndis;
            stats.ndis_first = ndis;
            stats.nhop_first = nhop;
            stats.nhop = nhop;
            stats.nfilter_first = nr_filter;
            stats.nfilter = nr_filter;
            return stats;
        }
    }
}

} // namespace

void PathSeer::add_links_starting_from_only_expanding(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        omp_lock_t* locks,
        VisitedTable& vt) {

    // expanding vecs, we store the nearest vector in heap top instead
    std::priority_queue<NodeDistCloser> other_link_targets;

    // step 1: search the neighbors for the expanding zone
    pathseer_search_neighbors_to_add_only_expanding(*this, ptdis, other_link_targets, nearest, d_nearest, level, vt);

    // step 2: add the points of other vecs
    std::vector<storage_idx_t> neighbors;
    neighbors.reserve(other_link_targets.size());

    // #if PATHSEER_CONSTRUCTION_METHOD>=2
    add_link_extending_batched(*this, ptdis, pt_id, other_link_targets, neighbors, level);
    // #else
    // while (!other_link_targets.empty()) {
    //     storage_idx_t other_id = other_link_targets.top().id;
    //     add_link_extending(*this, ptdis, pt_id, other_id, level);
    //     neighbors.push_back(other_id);
    //     other_link_targets.pop();
    // }
    // #endif

    omp_unset_lock(&locks[pt_id]);
    for (storage_idx_t other_id : neighbors) {
        omp_set_lock(&locks[other_id]);

        if(this->use_optimized_building)
            add_link_extending_fast(*this, ptdis, other_id, pt_id, level);
        else
            add_link_extending(*this, ptdis, other_id, pt_id, level);
        // #if PATHSEER_CONSTRUCTION_METHOD>=3
        // add_link_extending_fast(*this, ptdis, other_id, pt_id, level);
        // #else
        // add_link_extending(*this, ptdis, other_id, pt_id, level);
        // #endif
        omp_unset_lock(&locks[other_id]);
    }

    omp_set_lock(&locks[pt_id]);

}

void PathSeer::add_links_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        omp_lock_t* locks,
        VisitedTable& vt) {
    // dist-oriented vecs
    std::priority_queue<NodeDistCloser> link_targets;
    // expanding vecs, we store the nearest vector in heap top instead
    std::priority_queue<NodeDistCloser> other_link_targets;

    // step 1: separate two kinds of linked targets
    pathseer_search_neighbors_to_add(*this, ptdis, link_targets, other_link_targets, nearest, d_nearest, level, vt);

    // but we can afford only this many neighbors
    int neighbor_max = nb_neighbors(level);

    // step 2: applying different shrink methods
    // 2.1 shrink the link targets, and adding those that are pruned into other link targets
    int dist_boundary = level==0 ? 2*this->M : this->M;
    ::faiss::pathseer_shrink_neighbor_list(ptdis, link_targets, other_link_targets, dist_boundary);

    // 2.2 cut-off the redundant links
    // int dist_boundary = level==0 ? 2*this->M : this->M;
    // while(other_link_targets.size() > neighbor_max - dist_boundary)
    //     other_link_targets.pop();

    // 3. add the points of the dist_oriented ones
    std::vector<storage_idx_t> neighbors;
    neighbors.reserve(link_targets.size());
    // #if PATHSEER_CONSTRUCTION_METHOD>=2
    add_link_dist_oriented_batched(*this, ptdis, pt_id, link_targets, neighbors, other_link_targets, level);
    // #else
    // while (!link_targets.empty()) {
    //     storage_idx_t other_id = link_targets.top().id;
    //     add_link_dist_oriented(*this, ptdis, pt_id, other_id, level);
    //     neighbors.push_back(other_id);
    //     link_targets.pop();
    // }
    // #endif

    omp_unset_lock(&locks[pt_id]);
    for (storage_idx_t other_id : neighbors) {
        omp_set_lock(&locks[other_id]);
        // #if PATHSEER_CONSTRUCTION_METHOD>=2
        add_link_dist_oriented_opt(*this, ptdis, other_id, pt_id, level);
        // #else
        // add_link_dist_oriented(*this, ptdis, other_id, pt_id, level);
        // #endif
        omp_unset_lock(&locks[other_id]);
    }
    omp_set_lock(&locks[pt_id]);

    // 4. add the points of other vecs
    neighbors.clear();
    neighbors.reserve(other_link_targets.size());
    // #if PATHSEER_CONSTRUCTION_METHOD>=2
    add_link_extending_batched(*this, ptdis, pt_id, other_link_targets, neighbors, level);
    // #else
    // while (!other_link_targets.empty()) {
    //     storage_idx_t other_id = other_link_targets.top().id;
    //     add_link_extending(*this, ptdis, pt_id, other_id, level);
    //     neighbors.push_back(other_id);
    //     other_link_targets.pop();
    // }
    // #endif

    omp_unset_lock(&locks[pt_id]);
    for (storage_idx_t other_id : neighbors) {
        omp_set_lock(&locks[other_id]);
        if(this->use_optimized_building)
            add_link_extending_fast(*this, ptdis, other_id, pt_id, level);
        else
            add_link_extending(*this, ptdis, other_id, pt_id, level);
        // #if PATHSEER_CONSTRUCTION_METHOD>=3
        // add_link_extending_fast(*this, ptdis, other_id, pt_id, level);
        // #else
        // add_link_extending(*this, ptdis, other_id, pt_id, level);
        // #endif
        omp_unset_lock(&locks[other_id]);
    }

    omp_set_lock(&locks[pt_id]);

}


/**************************************************************
 * Building, parallel
 **************************************************************/
// mod compared to original hnsnw
void PathSeer::add_with_locks(
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt) {

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

    // for only expanding, we should update the M
    if(use_only_expanding_idx){
        for(int l = level; l>=0; l--){
            set_neighbor_dist_boundary(pt_id, level, (level==0) ? 2*M : M);
        }
    }

    for (; level > pt_level; level--) {
        greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
    }

    for (; level >= 0; level--) {
        if(use_only_expanding_idx)
            add_links_starting_from_only_expanding(
                    ptdis, pt_id, nearest, d_nearest, level, locks.data(), vt);
        else
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

using MinimaxHeap = PathSeer::MinimaxHeap;
using Node = PathSeer::Node;
using NeighNode = PathSeer::NeighNode;

/** Do a BFS on the candidates list */
// this function only search without filter
int search_from_candidates(
        const PathSeer& pathseer,
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        PathSeerStats& stats,
        int level,
        int nres_in = 0,
        const SearchParametersPathSeer* params = nullptr) {

    int nres = nres_in;
    int ndis = 0;

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : pathseer.check_relative_distance;
    int efSearch = params ? params->efSearch : pathseer.efSearch;

    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (nres < k) {
            faiss::maxheap_push(++nres, D, I, d, v1);
        } else if (d < D[0]) {
            faiss::maxheap_replace_top(nres, D, I, d, v1);
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) {

        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        size_t begin, end;
        pathseer.neighbor_range(v0, level, &begin, &end);
        end = begin + pathseer.get_neighbor_dist_boundary(v0, level);
        for (size_t j = begin; j < end; j++) {
            int v1 = pathseer.neighbors[j];
            if (v1 < 0)
                break;
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);
            ndis++;
            float d = qdis(v1);
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
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
        stats.nhop += nstep;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
    }

    return nres;
}

// functions used to extract vectors from dist_filter_queue to the result queue

// efficient fixed-size heap
struct HeapNode {
    idx_t id;
    float dist;
    HeapNode(idx_t id, float dist) : id(id), dist(dist) {}
    HeapNode() : id(-1), dist(0) {}
};

class MinHeap {
private:
    std::vector<HeapNode> heap;
    size_t heap_size;
    size_t max_size;

    void swap(size_t i, size_t j) {
        HeapNode temp = heap[i];
        heap[i] = heap[j];
        heap[j] = temp;
    }

    void heapifyUp(size_t index) {
        while (index > 0) {
            size_t parent = (index - 1) >> 1;
            if (heap[index].dist >= heap[parent].dist)
                break;
            swap(index, parent);
            index = parent;
        }
    }

    void heapifyDown(size_t index) {
        size_t left =  (index << 1) | 1;
        size_t right = (index + 1) << 1;
        size_t smallest = index;

        if (left < heap_size && heap[left].dist < heap[smallest].dist)
            smallest = left;
        if (right < heap_size && heap[right].dist < heap[smallest].dist)
            smallest = right;

        if (smallest != index) {
            swap(index, smallest);
            heapifyDown(smallest);
        }
    }

public:
    MinHeap(size_t max_size) : max_size(max_size), heap_size(0) {
        heap.resize(max_size);
    }

    const HeapNode& top() const {
        if (heap_size == 0) {
            throw std::out_of_range("Heap is empty.");
        }
        return heap.front();
    }

    void pop() {
        if (heap_size == 0) {
            throw std::out_of_range("Heap is empty.");
        }
        if(heap_size==1){
            heap_size = 0;
            return;
        }
        heap[0] = heap[heap_size-1];
        heap_size--;
        heapifyDown(0);
    }

    void push(idx_t id, float dist) {
        if (heap_size < max_size) {
            heap[heap_size++] = {id, dist};
            heapifyUp(heap_size - 1);
        } else if (dist < heap.front().dist) {
            heap[0] = {id, dist};
            heapifyDown(0);
        }
    }

    size_t size() const {
        return heap_size;
    }

    bool empty() const {
        return heap_size == 0;
    }

    void clear() {
        heap_size = 0;
    }

};

// The version with optimized filtering
int adaptive_hybrid_search_from_candidates(
        const PathSeer& pathseer,
        DistanceComputer& qdis,
        IDSelector* sel,
        int pos,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        PathSeerStats& stats,
        int level,
        int M_exp_search,
        int nres_in = 0,
        const SearchParametersPathSeer* params = nullptr) {

    int nres = nres_in;
    int ndis = 0;
    int nfilter = 0;
    int search_mode = 0; // 0: full-search, 1: only_perform_mff (one-hop), 2: acorn-based method, 3: only_perform_dfn
    if(params){
        if(params->only_perform_mff)
            search_mode = 1;
        else if(params->two_hop_search)
            search_mode = 2;
        else if(params->only_perform_dfn)
            search_mode = 3;
    }

    // after the profiling, M_exp_search cannot be zero here
    FAISS_ASSERT(M_exp_search >= 0 && M_exp_search <= 2*pathseer.M_exp);

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : pathseer.check_relative_distance;
    int efSearch = params ? params->efSearch : pathseer.efSearch;

    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        nfilter++;
        if(sel->is_member(v1, pos)){
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
            }
        }
        vt.set(v1);
    }

    int nstep = 0;
    int nr_inspect = M_exp_search;

    int* neigh_arr = new int[pathseer.M_exp];  // we should set a larger candidate array
    memset(neigh_arr, 0, sizeof(int) * nr_inspect);

    int this_M_exp_search = M_exp_search;

    // printf("[Query search mode] %d\n", search_mode);

    if(search_mode==2 || search_mode==3)
        printf("[Q]\n");

    while (candidates.size() > 0) {

        float d0 = 0;
        int v0;
        v0 = candidates.pop_min(&d0);

        // printf("[Front] %d, %f\n", v0, d0);

        if (do_dis_check) {
            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        if(search_mode==2 || search_mode==3)
            printf("[N]\n");

        int dist_boundary = pathseer.get_neighbor_dist_boundary(v0, level);

        int this_M_exp_search = M_exp_search;

        size_t begin, end;
        pathseer.neighbor_range(v0, level, &begin, &end);

        int neigh_min_ptr = 0;

        if(search_mode==0){
            // firstly, we must traverse towards the dist boundary, record each id
            for(size_t j=begin; j<begin+dist_boundary; j++){
                auto v1 = pathseer.neighbors[j];
                if(v1 < 0)
                    break;
                if(vt.get(v1)){
                    continue;
                }
                vt.set(v1);
                neigh_arr[neigh_min_ptr++] = v1;
            }

            // perform distance computation on them
            for(size_t j=0; j<neigh_min_ptr; j++){

                #ifdef ENABLE_PREFETCH
                if(neigh_min_ptr > j+1)
                    qdis.prefetch_vector(neigh_arr[j+1]);
                #endif

                auto v1 = neigh_arr[j];
                ndis++;
                float d = qdis(v1);
                candidates.push(v1, d);

                if(nres < k || d < D[0]){
                    nfilter++;
                    if (sel->is_member(v1, pos)){
                        // printf("[One for res]\n");
                        if (nres < k) {
                            faiss::maxheap_push(++nres, D, I, d, v1);
                        } else if (d < D[0]) {
                            faiss::maxheap_replace_top(nres, D, I, d, v1);
                        }
                    }
                }
            }
            neigh_min_ptr = 0;

            for(size_t j=begin+dist_boundary; j<begin+this_M_exp_search; j++){
                auto v1 = pathseer.neighbors[j];
                if(v1 < 0)
                    break;
                if(vt.get(v1)){
                    continue;
                }
                vt.set(v1);
                nfilter++;

                if (sel->is_member(v1, pos)){
                    FAISS_ASSERT(neigh_min_ptr < nr_inspect);
                    neigh_arr[neigh_min_ptr++] = v1;
                }
            }
        }
        else if(search_mode==1){
            for(size_t j=begin; j<begin+this_M_exp_search; j++){
                auto v1 = pathseer.neighbors[j];
                if(v1 < 0)
                    break;
                if(vt.get(v1)){
                    continue;
                }
                vt.set(v1);
                nfilter++;
                if (sel->is_member(v1, pos)){
                    FAISS_ASSERT(neigh_min_ptr < nr_inspect);
                    neigh_arr[neigh_min_ptr++] = v1;
                }
            }
        }
        else if(search_mode==2){
            bool keep_expanding = true;
            int num_found = 0;
            for(size_t j=begin; j<begin+this_M_exp_search; j++){
                auto v1 = pathseer.neighbors[j];
                if(v1 < 0)
                    break;

                bool pass_filter = false;
                nfilter++;

                if (sel->is_member(v1, pos)) {
                    pass_filter = true;
                    num_found++;
                }

                float d = qdis(v1);
                printf("[I] %f %d\n", d, pass_filter);

                if(vt.get(v1)){
                    continue;
                }

                if(pass_filter){
                    neigh_arr[neigh_min_ptr++] = v1;
                    vt.set(v1);
                    ndis++;
                    // we omit the dist-comp and leave it below
                    if (num_found >= nr_inspect) {
                        keep_expanding = false;
                        break;
                    }
                }

                // the two-hop search, unlike ACORN, we exploit a simple way to two-hop search all neighbors
                if(keep_expanding){
                    size_t begin2, end2;
                    pathseer.neighbor_range(v1, level, &begin2, &end2);
                    for (size_t j2 = begin2; j2 < end2; j2++) {
                        
                        auto v2 = pathseer.neighbors[j2];
                        if (v2 < 0) {
                            break;
                        }

                        nfilter++;
                        bool pass_filter2 = sel->is_member(v2, pos);

                        float d2 = qdis(v1);
                        printf("[I] %f %d\n", d2, pass_filter2);

                        if (pass_filter2) {
                            num_found++;
                        } else {
                            continue;
                        }

                        if (vt.get(v2)) {
                            continue;
                        }
                        vt.set(v2);

                        ndis++;

                        neigh_arr[neigh_min_ptr++] = v2;

                        if (num_found >= nr_inspect) {
                            keep_expanding = false;
                            break;
                        }
                    }
                }

            }
        }
        else if(search_mode==3){
            for(size_t j=begin; j<begin+dist_boundary; j++){
                auto v1 = pathseer.neighbors[j];
                if(v1 < 0)
                    break;
                if(vt.get(v1)){
                    continue;
                }
                vt.set(v1);
                neigh_arr[neigh_min_ptr++] = v1;
            }

            for(size_t j=0; j<neigh_min_ptr; j++){

                #ifdef ENABLE_PREFETCH
                if(neigh_min_ptr > j+1)
                    qdis.prefetch_vector(neigh_arr[j+1]);
                #endif

                auto v1 = neigh_arr[j];
                ndis++;
                float d = qdis(v1);
                candidates.push(v1, d);

                if(nres < k || d < D[0]){
                    nfilter++;
                    if (sel->is_member(v1, pos)){
                        printf("[I] %f 1\n", d);
                        if (nres < k) {
                            faiss::maxheap_push(++nres, D, I, d, v1);
                        } else if (d < D[0]) {
                            faiss::maxheap_replace_top(nres, D, I, d, v1);
                        }
                    }
                    else
                        printf("[I] %f 0\n", d);
                }
            }

        }
        else
            FAISS_ASSERT(false);

        int j=0;

        for(; j<neigh_min_ptr; j++){
            #ifdef ENABLE_PREFETCH
            if(neigh_min_ptr > j+1)
                qdis.prefetch_vector(neigh_arr[j+1]);
            #endif
            auto v1 = neigh_arr[j];
            ndis++;
            float d = qdis(v1);

            candidates.push(v1, d);

            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
            }
        }

        nstep++; 
        if (!do_dis_check && nstep > efSearch) {
            break;
        }

    }

    if (level == 0) {
        stats.n1++;
        stats.nhop += nstep;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nfilter += nfilter;

    }

    delete neigh_arr;
    return nres;
}

int adaptive_hybrid_search_from_candidates_print_overhead(
        const PathSeer& pathseer,
        DistanceComputer& qdis,
        IDSelector* sel,
        int pos,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        PathSeerStats& stats,
        int level,
        int M_exp_search,
        long long int dist_comp_ns,
        long long int filter_time_ns,
        bool print_virtual_overhead,
        int nres_in = 0,
        const SearchParametersPathSeer* params = nullptr) {

    int nres = nres_in;
    int ndis = 0;
    int nfilter = 0;
    size_t query_neigh_virtual_overhead = 0;
    int nr_query_neigh = 0;
    bool use_full_search = params ? !params->only_perform_mff : true;

    // after the profiling, M_exp_search cannot be zero here
    FAISS_ASSERT(M_exp_search >= 0 && M_exp_search <= 2*pathseer.M_exp);

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : pathseer.check_relative_distance;
    int efSearch = params ? params->efSearch : pathseer.efSearch;

    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        nfilter++;
        if(sel->is_member(v1, pos)){
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
            }
        }
        vt.set(v1);
    }

    int nstep = 0;
    int nr_inspect = std::max(M_exp_search, 2 * pathseer.M);

    int* neigh_arr = new int[nr_inspect];
    memset(neigh_arr, 0, sizeof(int) * nr_inspect);

    int this_M_exp_search = M_exp_search;

    // initial estimated single overhead
    long long int estimated_single_overhead = filter_time_ns;

    while (candidates.size() > 0) {

        float d0 = 0;
        int v0;
        v0 = candidates.pop_min(&d0);

        int num_visited = 0;
        int num_new = 0;
        long long int now_neigh_overhead = 0;

        if (do_dis_check) {
            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        int dist_boundary = pathseer.get_neighbor_dist_boundary(v0, level);

        int this_M_exp_search = M_exp_search;

        size_t begin, end;
        pathseer.neighbor_range(v0, level, &begin, &end);

        int neigh_min_ptr = 0;

        // counter for selectivity
        int nr_satisfacotry = 0;

        for(size_t j=begin; j<begin+dist_boundary; j++){
            auto v1 = pathseer.neighbors[j];
            if(v1 < 0)
                break;
            if(vt.get(v1)){
                num_visited++;
                continue;
            }
            vt.set(v1);
            neigh_arr[neigh_min_ptr++] = v1;
        }

        // perform distance computation on them
        for(size_t j=0; j<neigh_min_ptr; j++){
            #ifdef ENABLE_PREFETCH
            if(neigh_min_ptr > j+1)
                qdis.prefetch_vector(neigh_arr[j+1]);
            #endif
            auto v1 = neigh_arr[j];
            ndis++;
            float d = qdis(v1);
            candidates.push(v1, d);
            if(nres < k || d < D[0]){
                num_new++;
                nfilter++;
                now_neigh_overhead += filter_time_ns;
                if (sel->is_member(v1, pos)){
                    nr_satisfacotry++;
                    if (nres < k) {
                        faiss::maxheap_push(++nres, D, I, d, v1);
                    } else if (d < D[0]) {
                        faiss::maxheap_replace_top(nres, D, I, d, v1);
                    }
                }
            }
            else
                num_visited++;
        }

        neigh_min_ptr = 0;
        for(size_t j=begin+dist_boundary; j<begin+this_M_exp_search; j++){
            auto v1 = pathseer.neighbors[j];
            if(v1 < 0)
                break;
            if(vt.get(v1)){
                num_visited++;
                continue;
            }
            vt.set(v1);
            nfilter++;
            num_new++;
            if (sel->is_member(v1, pos)){
                FAISS_ASSERT(neigh_min_ptr < nr_inspect);
                neigh_arr[neigh_min_ptr++] = v1;
                nr_satisfacotry++;
            }
        }

        if(num_new){
            now_neigh_overhead = (num_new * filter_time_ns + nr_satisfacotry * dist_comp_ns) * (num_new + num_visited) / num_new;
        }
        else
            now_neigh_overhead = num_visited * filter_time_ns;

        if(print_virtual_overhead){
            #pragma omp critical
            printf("[O] %lld\n", now_neigh_overhead);
        }

        query_neigh_virtual_overhead += now_neigh_overhead;
        nr_query_neigh++;

        size_t j=0;
        for(; j<neigh_min_ptr; j++){
            #ifdef ENABLE_PREFETCH
            if(neigh_min_ptr > j+1)
                qdis.prefetch_vector(neigh_arr[j+1]);
            #endif
            auto v1 = neigh_arr[j];
            ndis++;
            float d = qdis(v1);

            candidates.push(v1, d);

            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
            }
        }

        nstep++; 
        if (!do_dis_check && nstep > efSearch) {
            break;
        }

    }

    if (level == 0) {
        stats.n1++;
        stats.nhop += nstep;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nfilter += nfilter;
        stats.virtual_overhead += (query_neigh_virtual_overhead / nr_query_neigh);
    }

    delete neigh_arr;
    return nres;
}


// a tentative optimized version
int adaptive_hybrid_search_from_candidates_overhead_bounded(
        const PathSeer& pathseer,
        DistanceComputer& qdis,
        IDSelector* sel,
        int pos,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        PathSeerStats& stats,
        int level,
        int M_exp_search,
        long long int dist_comp_ns,
        long long int filter_time_ns,
        int nres_in = 0,
        const SearchParametersPathSeer* params = nullptr) {

    int nres = nres_in;
    int ndis = 0;
    int nfilter = 0;
    bool use_full_search = params ? !params->only_perform_mff : true;
    // a global counter used when local Sp is not avaiable
    int nsatisfy = 0;

    // after the profiling, M_exp_search cannot be zero here
    FAISS_ASSERT(M_exp_search >= 0 && M_exp_search <= 2*pathseer.M_exp);
    FAISS_ASSERT(dist_comp_ns!=-1 && filter_time_ns!=-1);

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : pathseer.check_relative_distance;
    int efSearch = params ? params->efSearch : pathseer.efSearch;

    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        nfilter++;
        if(sel->is_member(v1, pos)){
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
            }
        }
        vt.set(v1);
    }

    int nstep = 0;
    int nr_inspect = std::max(M_exp_search, 2 * pathseer.M);

    int* neigh_arr = new int[nr_inspect];
    memset(neigh_arr, 0, sizeof(int) * nr_inspect);

    int this_M_exp_search = M_exp_search;

    long long int both_ops_overhead = dist_comp_ns + filter_time_ns;
    int nr_overhead_mul = pathseer.M * 2;
    long long int overhead_limit = nr_overhead_mul * both_ops_overhead;
    float coef_a = 0, coef_b = 0;

    bool vo_coef_mode = false;
    // if we have profiled overhead limit, use it
    if(params->coef_a && params->coef_b){
        coef_a = params->coef_a;
        coef_b = params->coef_b;
        vo_coef_mode = true;
    }
    // use the fixed virtual overhead if coef a and b is not present
    if(params && params->virtual_overhead)
        overhead_limit = params->virtual_overhead;

    // initial estimated single overhead
    long long int estimated_single_overhead = filter_time_ns;

    while (candidates.size() > 0) {

        float d0 = 0;
        int v0;
        v0 = candidates.pop_min(&d0);

        long long int now_neigh_overhead = 0;

        if (do_dis_check) {
            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        int dist_boundary = pathseer.get_neighbor_dist_boundary(v0, level);

        int this_M_exp_search = M_exp_search;

        int num_visited = 0;
        int num_new = 0;

        size_t begin, end;
        pathseer.neighbor_range(v0, level, &begin, &end);

        int neigh_min_ptr = 0;

        // counter for selectivity
        int nr_satisfacotry = 0;

        // method 1: fast but inaccurate
        // // firstly, we must traverse towards the dist boundary, record each id
        // for(int j=begin; j<begin+dist_boundary; j++){
        //     auto v1 = pathseer.neighbors[j];
        //     if(v1 < 0)
        //         break;
        //     if(vt.get(v1)){
        //         num_visited++;
        //         continue;
        //     }
        //     vt.set(v1);
        //     neigh_arr[neigh_min_ptr++] = v1;
        // }

        // // perform distance computation on them
        // for(int j=0; j<neigh_min_ptr; j++){

        //     #ifdef ENABLE_PREFETCH
        //     if(neigh_min_ptr > j+1)
        //         qdis.prefetch_vector(neigh_arr[j+1]);
        //     #endif

        //     auto v1 = neigh_arr[j];
        //     ndis++;
        //     float d = qdis(v1);
        //     candidates.push(v1, d);

        //     if(nres < k || d < D[0]){
        //         num_new++;
        //         nfilter++;
        //         now_neigh_overhead += filter_time_ns;
        //         if (sel->is_member(v1, pos)){
        //             now_neigh_overhead += dist_comp_ns;
        //             nr_satisfacotry++;
        //             if (nres < k) {
        //                 faiss::maxheap_push(++nres, D, I, d, v1);
        //             } else if (d < D[0]) {
        //                 faiss::maxheap_replace_top(nres, D, I, d, v1);
        //             }
        //         }
        //     }
        //     else
        //         num_visited++;
        // }

        // neigh_min_ptr = 0;

        // float estimated_local_selectivity = num_new ? 1.0 * nr_satisfacotry / num_new : -1;
        // // get the virtual overhead for this workload. If we have -1 selectivity, use the estimated average virtual overhead
        // long long int local_overhead_limit = overhead_limit;
        // if(vo_coef_mode && estimated_local_selectivity!=-1)
        //     local_overhead_limit = coef_a * estimated_local_selectivity + coef_b;

        // // the second stage, we should count the visited one
        // // we only update the estimated_single_overhead if we can update it. The first try must success as no vector is visited, so this variable must have a usable value.
        // if(num_new)
        //     estimated_single_overhead = now_neigh_overhead / num_new;
        // now_neigh_overhead += estimated_single_overhead * num_visited;

        // for(int j=begin+dist_boundary; j<begin+this_M_exp_search; j++){
        //     auto v1 = pathseer.neighbors[j];
        //     if(v1 < 0)
        //         break;
        //     if(vt.get(v1)){
        //         now_neigh_overhead += estimated_single_overhead;
        //         continue;
        //     }
        //     vt.set(v1);
        //     nfilter++;
        //     now_neigh_overhead += filter_time_ns;
        //     if (sel->is_member(v1, pos)){
        //         FAISS_ASSERT(neigh_min_ptr < nr_inspect);
        //         neigh_arr[neigh_min_ptr++] = v1;
        //         now_neigh_overhead += dist_comp_ns;
        //     }
        //     if(now_neigh_overhead >= local_overhead_limit){
        //         break;
        //     }
        // }

        // for(int j=0; j<neigh_min_ptr; j++){
        //     #ifdef ENABLE_PREFETCH
        //     if(neigh_min_ptr > j+1)
        //         qdis.prefetch_vector(neigh_arr[j+1]);
        //     #endif
        //     auto v1 = neigh_arr[j];
        //     ndis++;
        //     float d = qdis(v1);

        //     candidates.push(v1, d);

        //     if (nres < k) {
        //         faiss::maxheap_push(++nres, D, I, d, v1);
        //     } else if (d < D[0]) {
        //         faiss::maxheap_replace_top(nres, D, I, d, v1);
        //     }
        // }

        // int j = neigh_max_ptr + 1;

        // // dist-comp of non-filtered vectors
        // while(j < nr_inspect){
        //     #ifdef ENABLE_PREFETCH
        //     if(neigh_max_ptr > j+1)
        //         qdis.prefetch_vector(neigh_arr[j+1]);
        //     #endif
        //     auto v1 = neigh_arr[j];
        //     ndis++;
        //     float d = qdis(v1);
        //     candidates.push(v1, d);
        //     j++;
        // }

        // method 2: slow but accurate
        // step 1: traverse the sparse zone as we must do it
        for(size_t j=begin; j<begin+dist_boundary; j++){
            auto v1 = pathseer.neighbors[j];
            if(v1 < 0)
                break;
            if(vt.get(v1)){
                num_visited++;
                continue;
            }
            vt.set(v1);
            neigh_arr[neigh_min_ptr++] = v1;
        }

        // step 2: batch perform distance computation on unvisted vectors of the sparse zone
        for(size_t j=0; j<neigh_min_ptr; j++){
            #ifdef ENABLE_PREFETCH
            if(neigh_min_ptr > j+1)
                qdis.prefetch_vector(neigh_arr[j+1]);
            #endif
            auto v1 = neigh_arr[j];
            ndis++;
            float d = qdis(v1);
            candidates.push(v1, d);
            // only compute distance of vectors that are fairly near
            if(nres < k || d < D[0]){
                num_new++;
                nfilter++;
                if (sel->is_member(v1, pos)){
                    nr_satisfacotry++;
                    nsatisfy++;
                    if (nres < k) {
                        faiss::maxheap_push(++nres, D, I, d, v1);
                    } else if (d < D[0]) {
                        faiss::maxheap_replace_top(nres, D, I, d, v1);
                    }
                }
            }
            else
                num_visited++;
        }

        // step 3: pre-traverse the remaining vectors until stop condition is met
        neigh_min_ptr = 0;
        int cnt = 0;
        for(size_t j=begin+dist_boundary; j<begin+this_M_exp_search; j++){

            // we only judege every 5 times to reduce the overhead. We place it here because we may early stop for high-filtering-cost workload
            if(cnt % 5 == 0){
                // calculate the virtual overhead of now
                if(num_new){
                    now_neigh_overhead = (num_new * filter_time_ns + nr_satisfacotry * dist_comp_ns) * (num_new + num_visited) / num_new;
                }
                else{
                    // now_neigh_overhead = num_visited * filter_time_ns;
                    // estimate with workload Sp
                    now_neigh_overhead = num_visited * (filter_time_ns + 1.0 * nsatisfy / nfilter * dist_comp_ns);
                }

                // maybe we should calculate the overhead limit
                // long long int local_overhead_limit = (vo_coef_mode && num_new) ? (coef_a * 1.0 * nr_satisfacotry / num_new + coef_b) : overhead_limit;
                long long int local_overhead_limit = 0;
                if(vo_coef_mode){
                    local_overhead_limit = num_new ? (coef_a * 1.0 * nr_satisfacotry / num_new + coef_b) : (coef_a * 1.0 * nsatisfy / nfilter + coef_b);
                }
                else{
                    local_overhead_limit = overhead_limit;
                }

                if(now_neigh_overhead >= local_overhead_limit){
                    break;
                }
            }

            cnt++;

            auto v1 = pathseer.neighbors[j];
            if(v1 < 0)
                break;
            if(vt.get(v1)){
                num_visited++;
                continue;
            }
            vt.set(v1);
            nfilter++;
            num_new++;
            if (sel->is_member(v1, pos)){
                FAISS_ASSERT(neigh_min_ptr < nr_inspect);
                neigh_arr[neigh_min_ptr++] = v1;
                nr_satisfacotry++;
                nsatisfy++;
            }

        }

        for(size_t j=0; j<neigh_min_ptr; j++){
            #ifdef ENABLE_PREFETCH
            if(neigh_min_ptr > j+1)
                qdis.prefetch_vector(neigh_arr[j+1]);
            #endif
            auto v1 = neigh_arr[j];
            ndis++;
            float d = qdis(v1);

            candidates.push(v1, d);

            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
            }
        }

        nstep++; 
        if (!do_dis_check && nstep > efSearch) {
            break;
        }

    }

    if (level == 0) {
        stats.n1++;
        stats.nhop += nstep;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nfilter += nfilter;
    }

    // printf("[Query] filter_ns=%lld dist_ns=%lld\n", filter_time_ns, dist_comp_ns);

    delete neigh_arr;
    return nres;
}


} // anonymous namespace

PathSeerStats PathSeer::search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt,
        int pos,
        const SearchParametersPathSeer* params) const {

    PathSeerStats stats;
    if (entry_point == -1) {
        return stats;
    }

    if (upper_beam == 1) {
        IDSelector* sel = params->sel;
        // dist-only search
        if(!sel){
            storage_idx_t nearest = entry_point;
            float d_nearest = qdis(nearest);

            for (int level = max_level; level >= 1; level--) {
                PathSeerStats local_stats = greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
                stats.combine(local_stats);
            }

            int ef = std::max(params ? params->efSearch : efSearch, k);
            if (search_bounded_queue) {
                MinimaxHeap candidates(ef);
                candidates.push(nearest, d_nearest);
                search_from_candidates(
                        *this, qdis, k, I, D, candidates, vt, stats, 0, 0, params);
            } else {
                throw FaissException("UNIMPLEMENTED search unbounded queue");
                
            }
        }
        // metadata-oriented hybrid search
        else{
            int M_exp_search = params->M_exp_search;
            // if user set the M_exp_search, restrict it within [M, M_exp]
            if(M_exp_search!=0){
                M_exp_search = std::min<int>(std::max<int>(0, M_exp_search), this->M_exp);
            }

            bool search_with_overhead_bounded = (M_exp_search==0);

            //  greedy search on upper levels
            storage_idx_t nearest = entry_point;
            float d_nearest = qdis(nearest);
            long long int dist_comp_ns = profilier->dist_comp_cost_ns ? profilier->dist_comp_cost_ns : -1;
            long long int filter_time_ns = profilier->filter_cost_ns ? profilier->filter_cost_ns : -1;

            // printf("[entry] d_nearest=%f\n", d_nearest);

            for (int level = max_level; level >= 1; level--) {
                PathSeerStats local_stats = adaptive_hybrid_greedy_update_nearest(*this, qdis, sel, pos, level, nearest, d_nearest, M_exp_search, dist_comp_ns, filter_time_ns, search_with_overhead_bounded);
                stats.combine(local_stats);
                // printf("[%d] d_nearest=%f\n", level, d_nearest);
            }

            int ef = std::max(params ? params->efSearch : efSearch, k);

            if (search_bounded_queue) { // this is the most common branch
                MinimaxHeap candidates(ef);

                candidates.push(nearest, d_nearest);

                if(params && (params->profile_virtual_overhead || params->print_virutal_overhead))
                    adaptive_hybrid_search_from_candidates_print_overhead(
                            *this, qdis, sel, pos, k, I, D, candidates, vt, stats, 0, M_exp_search*2, dist_comp_ns, filter_time_ns, params->print_virutal_overhead, 0, params);
                else if(!search_with_overhead_bounded)
                    adaptive_hybrid_search_from_candidates(
                            *this, qdis, sel, pos, k, I, D, candidates, vt, stats, 0, M_exp_search*2, 0, params);
                else
                    adaptive_hybrid_search_from_candidates_overhead_bounded(
                            *this, qdis, sel, pos, k, I, D, candidates, vt, stats, 0, M_exp_search*2, dist_comp_ns, filter_time_ns, 0, params);
            } else {
                printf("UNIMPLEMENTED BRANCH for hybid search\n");
                throw FaissException("UNIMPLEMENTED search unbounded queue");
            }
        }

        vt.advance();

    } else
        throw FaissException("Currently-not-implemented branch");

    return stats;
}


/**************************************************************
 * MinimaxHeap
 **************************************************************/

// return 1 if any replacement takes place
int PathSeer::MinimaxHeap::push(storage_idx_t i, float v) {
    if (k == n) {
        if (v >= dis[0])
            return 0;
        faiss::heap_pop<HC>(k--, dis.data(), ids.data());
        --nvalid;
    }
    faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
    ++nvalid;
    return 1;
}

float PathSeer::MinimaxHeap::max() const {
    return dis[0];
}

int PathSeer::MinimaxHeap::size() const {
    return nvalid;
}

void PathSeer::MinimaxHeap::clear() {
    nvalid = k = 0;
}


#ifdef __AVX2__

int PathSeer::MinimaxHeap::pop_min(float* vmin_out) {
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
int PathSeer::MinimaxHeap::pop_min(float* vmin_out) {
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

int PathSeer::MinimaxHeap::count_below(float thresh) {
    int n_below = 0;
    for (int i = 0; i < k; i++) {
        if (dis[i] < thresh) {
            n_below++;
        }
    }

    return n_below;
}


} // namespace faiss
