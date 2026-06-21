#ifndef _NORMALIZE_HPP_
#define _NORMALIZE_HPP_
#include <cmath>  // For sqrt.

void normalize_L2(float* data, size_t d) {
    float norm = 0.0;
    for (size_t i = 0; i < d; ++i) {
        norm += data[i] * data[i];
    }
    norm = std::sqrt(norm);
    for (size_t i = 0; i < d; ++i) {
        data[i] /= norm;
    }
}

void normalize(float* data, size_t d, size_t n) {
    float* vector = data;
    for (size_t i = 0; i < n; ++i) {
        normalize_L2(vector, d);
        vector += d;
    }
}

#endif