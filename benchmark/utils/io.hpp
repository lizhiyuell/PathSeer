// Utility functions for file I/O.

#ifndef _IO_HPP_
#define _IO_HPP_

#include <iostream>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <cassert>
#include "normalize.hpp"
#include <typeinfo>

template <typename data_t>
void loadVectorData(const std::string& filename, uint32_t* n, uint32_t* d, data_t** data) {
    // Open the file in binary read mode.
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Read the number of vectors n.
    file.read(reinterpret_cast<char*>(n), sizeof(uint32_t));
    if (!file) throw std::runtime_error("Failed to read vector count (n)");

    // Read the vector dimension d.
    file.read(reinterpret_cast<char*>(d), sizeof(uint32_t));
    if (!file) throw std::runtime_error("Failed to read vector dimension (d)");

    // Calculate the total data size.
    size_t dataSize = static_cast<size_t>(*n) * static_cast<size_t>(*d);

    // Allocate memory for vector data.
    *data = new data_t[dataSize];
    if (!(*data)) throw std::runtime_error("Failed to allocate memory for data");

    // Read vector data.
    file.read(reinterpret_cast<char*>(*data), dataSize * sizeof(data_t));
    if (!file) throw std::runtime_error("Failed to read vector data");

    file.close();
}

// Load normalized vector data.
template <typename data_t>
void loadVectorDataNormalized(const std::string& filename, uint32_t* n, uint32_t* d, data_t** data) {

    loadVectorData<data_t>(filename, n, d, data);
    normalize(*data, *d, *n);

}

// Save two-dimensional data.

template <typename data_t>
void saveVectorData(const std::string& filename, uint32_t n, uint32_t d, const data_t* data) {
    // Open the binary file for writing.
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Write n and d as uint32_t values.
    outfile.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
    outfile.write(reinterpret_cast<const char*>(&d), sizeof(uint32_t));

    // Write the data array containing n*d data_t values.
    size_t data_size = static_cast<size_t>(n) * d;
    outfile.write(reinterpret_cast<const char*>(data), data_size * sizeof(data_t));

    if (!outfile) {
        throw std::runtime_error("Failed to write data to file: " + filename);
    }

    // Close the file.
    outfile.close();
}


// Template helper for loading one-dimensional data.
template <typename data_t>
void loadArray(const std::string& filename, uint32_t* n, data_t** data) {
    // Open the file in binary read mode.
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Read the number of values n.
    file.read(reinterpret_cast<char*>(n), sizeof(uint32_t));
    if (!file) throw std::runtime_error("Failed to read vector count (n)");

    uint32_t total_nr = static_cast<uint32_t>(*n);

    // Allocate memory for vector data.
    *data = new data_t[total_nr];
    if (!(*data)) throw std::runtime_error("Failed to allocate memory for data");

    // Read vector data.
    file.read(reinterpret_cast<char*>(*data), total_nr * sizeof(data_t));
    if (!file) throw std::runtime_error("Failed to read array data");

    file.close();
}

// Load one-dimensional data and build an external-to-internal mapping.
template <typename data_t>
std::unordered_map<data_t, data_t> loadLabelMapping(const std::string& filename) {

    std::unordered_map<data_t, data_t> result;

    uint32_t n;
    data_t* data;

    loadArray<data_t>(filename, &n, &data);

    for(int i=0; i<n; i++){
        result[data[i]] = i;
    }

    return result;

}

// Load a metadata file and return it as a string vector.
std::vector<std::string> load_metadata_string(std::string metadata_path){

    std::vector<std::string> result;
    std::ifstream infile(metadata_path);
    if(!infile){
        std::cerr << "Error opening metadata file: " << metadata_path << std::endl;
        assert(false);
    }
    std::string line;
    while (std::getline(infile, line)) {
        result.push_back(line);
    }

    return result;

}


#endif