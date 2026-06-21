#ifndef __SELECTOR_HPP__
#define __SELECTOR_HPP__
// self-defined selector functions

#include <string>
#include <cassert>
#include <faiss/impl/IDSelector.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>
#include <unordered_map>
#include <xmmintrin.h>
#include <random>
#include <cstdlib>
#include <cstddef>

// void sleep_for_ns(long long ns) {
//     auto start_time = std::chrono::high_resolution_clock::now();
//     auto end_time = start_time + std::chrono::nanoseconds(ns);

//     while (std::chrono::high_resolution_clock::now() < end_time) {
//     }
// }

void sleep_for_ns(long long ns) {
    float counter = 1;
    for(int i=0; i<ns; i++){
        counter *= 1.01;
    }
}

// using the low-cost bitmap and sleeping to simulate different filter costs
struct IDSelectorBitmapLatency : faiss::IDSelector {
    size_t n;
    const uint8_t* bitmap;
    int sleep_ns;

    /** Construct with a binary mask
     *
     * @param n size of the bitmap array
     * @param bitmap id will be selected iff id / 8 < n and bit number
     *               (i%8) of bitmap[floor(i / 8)] is 1.
     */
    IDSelectorBitmapLatency(size_t n, const uint8_t* bitmap) : n(n), bitmap(bitmap), sleep_ns(0) {}
    bool is_member(int64_t ii) const {
        uint64_t i = ii;
        if(sleep_ns)
            sleep_for_ns(sleep_ns);
        if ((i >> 3) >= n) {
            return false;
        }
        return (bitmap[i >> 3] >> (i & 7)) & 1;
    }

    void set_sleep_ns(int cycle_count){
        this->sleep_ns = cycle_count;
    }
};


// IDSelector that load text-based metadata and build the filter. Each number is an integer
struct IDSelectorIntArray : faiss::IDSelector {

    std::vector<int> metadata;
    
    // single filter or batch filter
    int filter;
    std::vector<int> query_filter;

    IDSelectorIntArray(std::string metadata_file){

        std::ifstream infile(metadata_file);
        if(!infile){
            std::cerr << "Error opening file: " << metadata_file << std::endl;
            assert(false);
        }
        std::string line;
        while (std::getline(infile, line)) {
            try {
                int num = std::stoi(line);  // Convert the string to an integer.
                metadata.push_back(num);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid integer in file: " << line << std::endl;
                assert(false);
            }
        }
        infile.close();

    }

    IDSelectorIntArray(std::vector<int> metadata_arr){

        this->metadata = metadata_arr;

    }

    // an artificial construction function, which randomly set integer for each metadata
    IDSelectorIntArray(int nr_vector){

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(1, 100);

        for(int i=0; i<nr_vector; i++)
            metadata.push_back(dist(gen));

    }

    void set_filter(std::string metadata){
        this->filter = std::stoi(metadata);
    }

    bool is_member(int64_t ii) const {
        return metadata[ii]==this->filter;
    }

    // ii: dataset idx, pos: query idx
    bool is_member(int64_t ii, int pos) const {
        return metadata[ii]==this->query_filter[pos];
    }

    void set_filter_batch(std::vector<std::string>& batch_filter){

        query_filter.reserve(batch_filter.size());
        for(auto it : batch_filter){
            query_filter.push_back(std::stoi(it));
        }

    }

};


// Same as the IDSelectorIntArray, but allowing setting a number for sleeping
struct IDSelectorIntArraySleep : faiss::IDSelector {

    std::vector<int> metadata;
    int sleep_ns;

    // single filter or batch filter
    int filter;
    std::vector<int> query_filter;

    IDSelectorIntArraySleep(std::string metadata_file){

        std::ifstream infile(metadata_file);
        if(!infile){
            std::cerr << "Error opening file: " << metadata_file << std::endl;
            assert(false);
        }
        std::string line;
        while (std::getline(infile, line)) {
            try {
                int num = std::stoi(line);  // Convert the string to an integer.
                metadata.push_back(num);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid integer in file: " << line << std::endl;
                assert(false);
            }
        }
        infile.close();

        this->sleep_ns = 0;

    }

    IDSelectorIntArraySleep(std::vector<int> metadata_arr){

        this->metadata = metadata_arr;
        this->sleep_ns = 0;

    }

    // an artificial construction function, which randomly set integer for each metadata
    IDSelectorIntArraySleep(int nr_vector){

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(1, 100);

        for(int i=0; i<nr_vector; i++)
            metadata.push_back(dist(gen));

        this->sleep_ns = 0;

    }

    void set_filter(std::string metadata){
        this->filter = std::stoi(metadata);
    }

    bool is_member(int64_t ii) const {
        sleep_for_ns(sleep_ns);
        return metadata[ii]==this->filter;
    }

    // ii: dataset idx, pos: query idx
    bool is_member(int64_t ii, int pos) const {
        sleep_for_ns(sleep_ns);
        return metadata[ii]==this->query_filter[pos];
    }

    void set_sleep_ns(int ns){
        this->sleep_ns = ns;
    }

    void set_filter_batch(std::vector<std::string>& batch_filter){

        query_filter.reserve(batch_filter.size());
        for(auto it : batch_filter){
            query_filter.push_back(std::stoi(it));
        }
    }
};


static std::vector<int> split_input(std::string input){
    
    std::stringstream a_stream(input);
    std::string a_part;
    std::vector<int> result;
    while (std::getline(a_stream, a_part, ',')) {
        if(a_part=="")
            result.push_back(-1);
        else
            result.push_back(atoi(a_part.c_str()));
    }
    if (input.back() == ',') {
        result.push_back(-1);
    }

    return result;
}


// containing multiple-attributes, sepearated with ",", int format
struct IDSelectorMultiAttribute : faiss::IDSelector {

    std::vector<std::vector<int>> metadata;
    
    // single filter or batch filter
    std::vector<int> filter;
    std::vector<std::vector<int>> query_filter;

    IDSelectorMultiAttribute(std::string metadata_file){

        std::ifstream infile(metadata_file);
        if(!infile){
            std::cerr << "Error opening file: " << metadata_file << std::endl;
            assert(false);
        }
        std::string line;
        while (std::getline(infile, line)) {
            std::vector<int> res = split_input(line);
            metadata.push_back(res);
        }
        infile.close();
    }

    void set_filter(std::string metadata){
        filter = split_input(metadata);
    }

    bool is_member(int64_t ii) const {

        for(int i=0; i<filter.size(); i++){
            if(filter[i]==-1)
                continue;
            if(filter[i] != metadata[ii][i])
                return false;
        }
        return true;
    }

    // ii: dataset idx, pos: query idx
    bool is_member(int64_t ii, int pos) const {

        assert(pos<query_filter.size());

        for(int i=0; i<query_filter[0].size(); i++){
            if(query_filter[pos][i]==-1)
                continue;
            if(query_filter[pos][i] != metadata[ii][i])
                return false;
        }

        return true;
    }


    void set_filter_batch(std::vector<std::string>& batch_filter){

        query_filter.reserve(batch_filter.size());
        for(auto it : batch_filter){
            query_filter.push_back(split_input(it));
        }
    }

};

// containing multiple-attributes, and support two kinds of operations
struct IDSelectorCompositeFiltering : faiss::IDSelector {

    std::vector<std::vector<int>> metadata;
    
    // single filter or batch filter
    std::vector<int> filter;
    std::vector<std::vector<int>> query_filter;

    IDSelectorCompositeFiltering(std::string metadata_file){

        std::ifstream infile(metadata_file);
        if(!infile){
            std::cerr << "Error opening file: " << metadata_file << std::endl;
            assert(false);
        }
        std::string line;
        while (std::getline(infile, line)) {
            std::vector<int> res = split_input(line);
            metadata.push_back(res);
        }
        infile.close();
    }

    void set_filter(std::string metadata){
        filter = split_input(metadata);
    }

    // the first number: find if the target is newer than the query, the remaining number find if there is an overlap
    bool is_member(int64_t ii) const {

        if(filter[0] > metadata[ii][0])
            return false;

        // the attribute has been sorted, just judge whether there is a match
        int a = 1, b = 1;
        while(a < filter.size() && b < metadata[ii].size()){
            if(filter[a] == metadata[ii][b])
                return true;
            if(filter[a] < metadata[ii][b])
                a++;
            else
                b++;
        }
        return false;
    }

    // ii: dataset idx, pos: query idx
    bool is_member(int64_t ii, int pos) const {

        if(query_filter[pos][0] > metadata[ii][0])
            return false;

        // the attribute has been sorted, just judge whether there is a match
        int a = 1, b = 1;
        while(a < query_filter[pos].size() && b < metadata[ii].size()){
            if(query_filter[pos][a] == metadata[ii][b])
                return true;
            if(query_filter[pos][a] < metadata[ii][b])
                a++;
            else
                b++;
        }
        return false;

    }


    void set_filter_batch(std::vector<std::string>& batch_filter){

        query_filter.reserve(batch_filter.size());
        for(auto it : batch_filter){
            query_filter.push_back(split_input(it));
        }
    }

};

// filter with string matching
struct IDSelectorRegex : faiss::IDSelector {

    std::vector<std::string> metadata;
    
    // single filter or batch filter
    std::string pattern;
    std::vector<std::string> query_filter;

    IDSelectorRegex(std::string metadata_file){

        std::ifstream infile(metadata_file);
        if(!infile){
            std::cerr << "Error opening file: " << metadata_file << std::endl;
            assert(false);
        }
        std::string line;
        while (std::getline(infile, line)) {
            metadata.push_back(line);
        }
        infile.close();

    }

    void set_filter(std::string metadata){
        this->pattern = metadata;
    }

    void set_filter_batch(std::vector<std::string>& batch_filter){

        query_filter.reserve(batch_filter.size());
        for(auto it : batch_filter){
            query_filter.push_back(it);
        }
    }

    bool is_member(int64_t ii) const {
        return metadata[ii].find(pattern) != std::string::npos;
    }

    // ii: dataset idx, pos: query idx
    bool is_member(int64_t ii, int pos) const {
       return metadata[ii].find(query_filter[pos]) != std::string::npos;
    }

};

static inline bool match_with_one_edit(const std::string& text, std::size_t i, const std::string& pat) {
    const std::size_t n = text.size();
    const std::size_t m = pat.size();

    std::size_t t = i;
    std::size_t p = 0;

    // Consume the common prefix.
    while (t < n && p < m && text[t] == pat[p]) {
        ++t; ++p;
    }

    // pattern is exhausted, so the distance is 0; extra text can be treated as another start point.
    if (p == m) return true;

    // text is exhausted; the remaining pattern can be at most one character for one edit.
    if (t == n) return (m - p) <= 1;

    // Try the three one-edit cases: replace, insert (skip text[t]), and delete (skip pat[p]).
    {
        std::size_t tt = t + 1, pp = p + 1;              // Replace.
        while (tt < n && pp < m && text[tt] == pat[pp]) { ++tt; ++pp; }
        if (pp == m) return true;
    }
    {
        std::size_t tt = t + 1, pp = p;                  // Insert; text has one extra character.
        while (tt < n && pp < m && text[tt] == pat[pp]) { ++tt; ++pp; }
        if (pp == m) return true;
    }
    {
        std::size_t tt = t, pp = p + 1;                  // Delete; text has one fewer character.
        while (tt < n && pp < m && text[tt] == pat[pp]) { ++tt; ++pp; }
        if (pp == m) return true;
    }

    return false;
}

// filter with fuzzy filter
struct IDSelectorFuzzyFilter : faiss::IDSelector {

    std::vector<std::string> metadata;
    
    // single filter or batch filter
    std::string pattern;
    std::vector<std::string> query_filter;

    IDSelectorFuzzyFilter(std::string metadata_file){

        std::ifstream infile(metadata_file);
        if(!infile){
            std::cerr << "Error opening file: " << metadata_file << std::endl;
            assert(false);
        }
        std::string line;
        while (std::getline(infile, line)) {
            metadata.push_back(line);
        }
        infile.close();

    }

    void set_filter(std::string metadata){
        this->pattern = metadata;
    }

    void set_filter_batch(std::vector<std::string>& batch_filter){

        query_filter.reserve(batch_filter.size());
        for(auto it : batch_filter){
            query_filter.push_back(it);
        }
    }

    bool is_member(int64_t ii) const {

        const std::size_t n = metadata[ii].size();
        const std::size_t m = pattern.size();

        if (m == 0) return true;
        if (n == 0) return false;

        // Fast path for exact matches.
        if (n >= m && metadata[ii].find(pattern) != std::string::npos) return true;

        // Scan start positions with early-stop pruning.
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t remain = n - i;
            if (remain + 1 < m) break; // Even one insertion cannot fit the pattern.
            if (match_with_one_edit(metadata[ii], i, pattern)) return true;
        }
        return false;

    }

    // ii: dataset idx, pos: query idx
    bool is_member(int64_t ii, int pos) const {

        const std::size_t n = metadata[ii].size();
        const std::size_t m = query_filter[pos].size();

        if (m == 0) return true;
        if (n == 0) return false;

        // Fast path for exact matches.
        if (n >= m && metadata[ii].find(query_filter[pos]) != std::string::npos) return true;

        // Scan start positions with early-stop pruning.
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t remain = n - i;
            if (remain + 1 < m) break; // Even one insertion cannot fit the pattern.
            if (match_with_one_edit(metadata[ii], i, query_filter[pos])) return true;
        }
        return false;

    }

};

// IDSelector that find the dataset int larger than the query int
struct IDSelectorIntArrayLarger : faiss::IDSelector {

    std::vector<int> metadata;
    
    // single filter or batch filter
    int filter;
    std::vector<int> query_filter;

    IDSelectorIntArrayLarger(std::string metadata_file){

        std::ifstream infile(metadata_file);
        if(!infile){
            std::cerr << "Error opening file: " << metadata_file << std::endl;
            assert(false);
        }
        std::string line;
        while (std::getline(infile, line)) {
            try {
                int num = std::stoi(line);  // Convert the string to an integer.
                metadata.push_back(num);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid integer in file: " << line << std::endl;
                assert(false);
            }
        }
        infile.close();

    }

    IDSelectorIntArrayLarger(std::vector<int> metadata_arr){

        this->metadata = metadata_arr;

    }

    // an artificial construction function, which randomly set integer for each metadata
    IDSelectorIntArrayLarger(int nr_vector){

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(1, 100);

        for(int i=0; i<nr_vector; i++)
            metadata.push_back(dist(gen));

    }

    void set_filter(std::string metadata){
        this->filter = std::stoi(metadata);
    }

    bool is_member(int64_t ii) const {
        return metadata[ii] > this->filter;
    }

    // ii: dataset idx, pos: query idx
    bool is_member(int64_t ii, int pos) const {
        return metadata[ii] > this->query_filter[pos];
    }

    void set_filter_batch(std::vector<std::string>& batch_filter){

        query_filter.reserve(batch_filter.size());
        for(auto it : batch_filter){
            query_filter.push_back(std::stoi(it));
        }

    }

};

// IDSelector that find the dataset int larger than the query int
struct IDSelectorFloatArrayLower : faiss::IDSelector {

    std::vector<float> metadata;
    
    // single filter or batch filter
    float filter;
    std::vector<float> query_filter;

    IDSelectorFloatArrayLower(std::string metadata_file){

        std::ifstream infile(metadata_file);
        if(!infile){
            std::cerr << "Error opening file: " << metadata_file << std::endl;
            assert(false);
        }
        std::string line;
        while (std::getline(infile, line)) {
            try {
                float num = std::stof(line);  // Convert the string to a float.
                metadata.push_back(num);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid integer in file: " << line << std::endl;
                assert(false);
            }
        }
        infile.close();

    }

    IDSelectorFloatArrayLower(std::vector<float> metadata_arr){

        this->metadata = metadata_arr;

    }

    void set_filter(std::string metadata){
        this->filter = std::stof(metadata);
    }

    bool is_member(int64_t ii) const {
        return metadata[ii] < this->filter;
    }

    // ii: dataset idx, pos: query idx
    bool is_member(int64_t ii, int pos) const {
        return metadata[ii] < this->query_filter[pos];
    }

    void set_filter_batch(std::vector<std::string>& batch_filter){

        query_filter.reserve(batch_filter.size());
        for(auto it : batch_filter){
            query_filter.push_back(std::stof(it));
        }

    }

};

#endif