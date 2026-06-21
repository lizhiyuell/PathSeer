// Load dataset and query data, including their class_int metadata.

#ifndef _LOAD_HPP_
#define _LOAD_HPP_

#include "io.hpp"
#include <string>
#include <cstring>

// Load only vector data, not metadata.
void load_ann_data(std::string dataset_dir, float** vector_data, uint32_t* nr_vector, uint32_t* dimension, float** query_data, uint32_t* nr_query, bool normalized=true){

    std::string file_path = dataset_dir + "/load/load.bin";

    // Load vector data.
    if(normalized){
        loadVectorDataNormalized<float>(file_path, nr_vector, dimension, vector_data);
    }
    else{
        loadVectorData<float>(file_path, nr_vector, dimension, vector_data);
    }

    file_path = dataset_dir + "/query/query.bin";

    // Load query data.
    if(normalized){
        loadVectorDataNormalized<float>(file_path, nr_query, dimension, query_data);
    }
    else{
        loadVectorData<float>(file_path, nr_query, dimension, query_data);
    }

    return;

}

// Load only metadata, not vector data.
void load_ann_metadata_bv_tight(std::string dataset_dir, std::string& metadata_class, uint32_t* nr_query, uint32_t* nr_bytes, char** query_bit_vector){

    // Load the bit vector for query metadata.
    std::string file_path = dataset_dir + "/query/query_" + metadata_class + ".bv";

    loadVectorData<char>(file_path, nr_query, nr_bytes, query_bit_vector);

}

void load_ann_metadata_bv(std::string dataset_dir, uint32_t nr_vector, std::string& metadata_class, uint32_t* nr_query, char** query_bit_vector){

    uint32_t nr_bytes;
    char* bit_vector_tight;
    std::string file_path = dataset_dir + "/query/query_" + metadata_class + ".bv";
    loadVectorData<char>(file_path, nr_query, &nr_bytes, &bit_vector_tight);

    // Expand the bit vector.
    size_t bit_array_length = nr_vector * (*nr_query);
    *query_bit_vector = (char*) new char[bit_array_length];
    memset(*query_bit_vector, 0, bit_array_length);

    // Each query has a separate bit vector, so rounding must be handled.

    int idx = 0;
    for(int i=0; i<(*nr_query); i++){
        for(int j=0; j<nr_vector; j++){
            int pos = i * (nr_bytes << 3) + j;
            int bv_idx = pos >> 3;
            int bv_off = pos & 7;
            if(bit_vector_tight[bv_idx] & (1 << bv_off)){
                (*query_bit_vector)[idx++] = 1;
            }
            else
                (*query_bit_vector)[idx++] = 0;
        }
    }

    delete bit_vector_tight;

}

// Combined helper that loads both data and metadata.
void load_ann_data_bv_tight(std::string dataset_dir, std::string& metadata_class, float** vector_data, uint32_t* nr_vector, uint32_t* dimension, float** query_data, uint32_t* nr_query, uint32_t* nr_bytes, char** query_bit_vector, bool normalized=true){

    load_ann_data(dataset_dir, vector_data, nr_vector, dimension, query_data, nr_query, normalized);

    load_ann_metadata_bv_tight(dataset_dir, metadata_class, nr_query, nr_bytes, query_bit_vector);
    
    return;

}

// Load data and query metadata for bit-vector filters. base_dir points to the dataset; query_bit_vector is returned as a char array.
void load_ann_data_bv(std::string dataset_dir, std::string& metadata_class, float** vector_data, uint32_t* nr_vector, uint32_t* dimension, float** query_data, uint32_t* nr_query, char** query_bit_vector, bool normalized=true){

    load_ann_data(dataset_dir, vector_data, nr_vector, dimension, query_data, nr_query, normalized);

    load_ann_metadata_bv(dataset_dir, *nr_vector, metadata_class, nr_query, query_bit_vector);

    return;

}

#endif