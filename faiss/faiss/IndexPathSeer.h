/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/PathSeer.h>
#include <faiss/utils/utils.h>

// added
#include <sys/time.h>
#include <stdio.h>
#include <iostream>

namespace faiss {

struct IndexPathSeer;

extern PathSeerStats pathseer_stats;

/** The PathSeer index is a normal random-access index with a PathSeer
 * link structure built on top */

struct IndexPathSeer : Index {
    typedef PathSeer::storage_idx_t storage_idx_t;

    // the link strcuture
    PathSeer pathseer;

    // the sequential storage
    bool own_fields;
    Index* storage;

    explicit IndexPathSeer(int d, int M, int M_exp, MetricType metric = METRIC_L2);
    explicit IndexPathSeer(Index* storage, int M, int M_exp);
//     explicit IndexPathSeer(); // TODO check this is right

    ~IndexPathSeer() override;

    // add n vectors of dimension d to the index, x is the matrix of vectors
    void add(idx_t n, const float* x) override;

    /// Trains the storage if needed
    void train(idx_t n, const float* x) override;

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    // get the graph distance between entry point and the golden in a certain layer
    int print_entry_distance(
            const float* x,
            IDSelector* sel,
            storage_idx_t golden_id,
            int layer,
            int &max_hop) const;

    void reconstruct(idx_t key, float* recons) const override;

    void reset() override;

    // functions related to dist-comp cost profiling
    void profile_dist_comp_cost();
    void profile_filter_cost(IDSelector* sel);
    void set_profiled_dist_comp_ns(long long int ns);
    void set_profiled_filter_ns(long long int ns);
    long long int get_profiled_dist_comp_ns();
    long long int get_profiled_filter_ns();

    void set_using_only_expanding_idx(bool val) { pathseer.use_only_expanding_idx = val; }
    bool get_using_only_expanding_idx() { return pathseer.use_only_expanding_idx; }

    void set_using_optimized_building(bool val) { pathseer.use_optimized_building = val; }
    bool get_using_optimized_building() { return pathseer.use_optimized_building; }

};

/** Flat index topped with with a PathSeer structure to access elements
 *  more efficiently.
 */

struct IndexPathSeerFlat : IndexPathSeer {
    IndexPathSeerFlat();
    IndexPathSeerFlat(int d, int M, int M_exp, MetricType metric = METRIC_L2);

};

} // namespace faiss
