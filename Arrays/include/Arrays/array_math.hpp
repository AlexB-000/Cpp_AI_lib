#pragma once
#include "array.hpp"

template<typename T>
Array<T> matrix_mult(const Array<T>& t1, const Array<T>& t2){
    if (t1.shape[1] != t2.shape[0]) {
        throw std::invalid_argument("Number of columns in the first matrix must match the number of rows in the second matrix.");
    }

    const uint32_t m = t1.shape[0];
    const uint32_t n = t1.shape[1];
    const uint32_t p = t2.shape[1];

    Array<T> result(std::vector<uint32_t>{m, p});

    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t k = 0; k < n; ++k) {
            for (uint32_t j = 0; j < p; ++j) {
                result.data[i * p + j] += t1.data[i * n + k] * t2.data[k * p + j];
            }
        }
    }
    return result;
}

template<typename T>
T vector_mult(const Array<T>& t1, const Array<T>& t2){
    if (t1.shape != t2.shape) {
        throw std::invalid_argument("vectors must have the same shape for dot product");
    }

    T sum = T();
    for (uint32_t i = 0; i < t1.data.size(); ++i) {
        sum += t1.data[i] * t2.data[i];
    }
    return sum;
}
template<typename T>
Array<T> matrix_vector_mult(const Array<T>& t1, const Array<T>& t2){
    if (t1.shape[1] != t2.shape[0]) {
        throw std::invalid_argument("Number of columns in the matrix must match the length of the vector.");
    }

    const uint32_t m = t1.shape[0];
    const uint32_t n = t1.shape[1];

    Array<T> result(std::vector<uint32_t>{m}, 0.0f);

    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            result.data[i] += t1.data[i * n + j] * t2.data[j];
        }
    }
    return result;
}

template<typename T>
Array<T> vector_matrix_mult(const Array<T>& t1, const Array<T>& t2){
    if (t1.shape[0] != t2.shape[0]) {
        throw std::invalid_argument("Length of the vector must match the number of lines in the matrix.");
    }
    const uint32_t m = t2.shape[0];
    const uint32_t n = t2.shape[1];
    
    Array<T> result(std::vector<uint32_t>{n}, 0.0f);
    
    for (uint32_t j = 0; j < m; ++j) {
        for (uint32_t k = 0; k < n; ++k) {
            result.data[k] += t1.data[j] * t2.data[j * n + k];
        }
    }
    return result;
}

template<typename T>
Array<T> matmul(const Array<T>& t1, const Array<T>& t2){
    if (t1.dim == 2 && t2.dim == 2) {
        return matrix_mult(t1, t2);
    }
    else if (t1.dim == 2 && t2.dim == 1){
        return matrix_vector_mult(t1, t2);
    }
    else if (t1.dim == 1 && t2.dim == 2){
        return vector_matrix_mult(t1, t2);
    }
    else if (t1.dim == 1 && t2.dim == 1){
        return vector_mult(t1, t2);
    }
    else {
        throw std::invalid_argument("Only 1D and 2D arrays are supported for matmul.");
    }
}