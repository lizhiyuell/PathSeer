#ifndef _RECALL_HPP_
#define _RECALL_HPP_

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <cassert>
#include "io.hpp"

template <typename label_t>
class Recall {
private:
    uint32_t n;                // Number of queries
    uint32_t topk;             // Top-K value
    uint32_t* groundTruth;  // Stores the top-k ground truth for each query

public:
    // Constructor
    Recall() : n(0), topk(0), groundTruth(nullptr) {}
    ~Recall() { delete groundTruth; }

    // Set metadata from a file
    void set_metadata(const std::string& filename) {
        if(groundTruth){
            delete groundTruth;
            groundTruth = nullptr;
        }
        loadVectorData<uint32_t>(filename, &n, &topk, &groundTruth);
    }

    // directly load ground true from faiss array
    void set_metadata(std::vector<int64_t>& label_arr, int top_k){

        this->topk = top_k;
        assert(label_arr.size() % top_k == 0);
        this->n = label_arr.size() / top_k;

        groundTruth = new uint32_t[label_arr.size()];
        for(int i=0; i<label_arr.size(); i++)
            groundTruth[i] = label_arr[i];

    }


    // Calculate recall for a specific query index
    double get_recall(int query_index, const label_t* retrieved, int set_topk=0) {
        if (query_index < 0 || query_index >= n) {
            throw std::out_of_range("Query index out of range.");
        }

        if(!set_topk)
            set_topk = topk;

        const uint32_t* ground_truth_start = &groundTruth[query_index * topk];
        std::unordered_set<label_t> ground_truth_set(ground_truth_start, ground_truth_start + set_topk);

        // make sure that there is no repeated vectors in the retrieved
        std::unordered_set<label_t> visited_index;

        int correct_count = 0;
        for (int i = 0; i < set_topk; ++i) {
            label_t retrieved_idx = retrieved[i];
            if(retrieved_idx!=-1 && visited_index.find(retrieved_idx)!=visited_index.end()){
                printf("Error: repeated vector found in result!\n");
                assert(false);
            }
            visited_index.insert(retrieved_idx);
            if (ground_truth_set.find(retrieved_idx) != ground_truth_set.end()) {
                ++correct_count;
            }
        }

        return static_cast<double>(correct_count) / set_topk;
    }

    // Get recall of an array of result in batch, and return the avg recall
    double get_recall_batch(int nr_query, const label_t* retrieved, int set_topk=0){

        if (nr_query <= 0 || nr_query > n) {
            throw std::out_of_range("Query number out of range.");
        }

        if(!set_topk)
            set_topk = topk;

        float recall = 0;

        for(int query_index=0; query_index<nr_query; query_index++){
            const uint32_t* ground_truth_start = &groundTruth[query_index * topk];

            std::unordered_set<label_t> ground_truth_set(ground_truth_start, ground_truth_start + set_topk);

            // make sure that there is no repeated vectors in the retrieved
            std::unordered_set<label_t> visited_index;

            int correct_count = 0;
            for (int i = 0; i < set_topk; ++i) {
                label_t retrieved_idx = retrieved[set_topk * query_index + i];
                if(retrieved_idx!=-1 && visited_index.find(retrieved_idx)!=visited_index.end()){
                    printf("Error: repeated vector found in result!\n");
                    assert(false);
                }
                visited_index.insert(retrieved_idx);
                if (ground_truth_set.find(retrieved_idx) != ground_truth_set.end()) {
                    ++correct_count;
                }
            }

            float this_recall = static_cast<double>(correct_count) / set_topk;

            recall += this_recall; 

        }

        return recall / nr_query;

    }

    // get the storage idx of the order-st closetest golden vector
    label_t get_golden_id(int query_index, int order){

        assert(order < topk);

        const uint32_t* ground_truth_start = &groundTruth[query_index * topk];

        return ground_truth_start[order];

    }

    // Clear internal state
    void clear() {
        n = 0;
        topk = 0;
        if(groundTruth){
            delete groundTruth;
            groundTruth = nullptr;
        }
    }
};

#endif