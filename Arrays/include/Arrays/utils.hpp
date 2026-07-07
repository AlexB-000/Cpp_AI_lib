#pragma once
#include "array.hpp"

template<typename T>
Array<T> max(const Array<T>& arr, const std::vector<uint32_t>& axes = {}) {
    std::vector<uint32_t> idx(arr.dim, 0);
    Array<T> max_val = arr.at(idx);
    while (true){
        uint32_t flat_index = arr.offset;
        for (size_t i = 0; i < arr.dim; ++i) {
            flat_index += idx[i] * arr.strides[i];
        }
        if ((*arr.data_ptr)[flat_index] > max_val.at(0)) {
            max_val = (*arr.data_ptr)[flat_index];
        }
        /* Increment the multi-dimensional index */
        for (int32_t i = arr.dim - 1; i >= 0; --i) {
            if (++idx[i] < arr.shape[i]) {
                break;
            }
            idx[i] = 0;
            if (i == 0) {
                return max_val;
            }
        }
    }
}

template<typename T>
Array<T> min(const Array<T>& arr, const std::vector<uint32_t>& axes = {}) {
    std::vector<uint32_t> idx(arr.dim, 0);
    Array<T> min_val = arr.at(idx);
    while (true){
        uint32_t flat_index = arr.offset;
        for (size_t i = 0; i < arr.dim; ++i) {
            flat_index += idx[i] * arr.strides[i];
        }
        if ((*arr.data_ptr)[flat_index] < min_val.at(0)) {
            min_val = (*arr.data_ptr)[flat_index];
        }
        /* Increment the multi-dimensional index */
        for (int32_t i = arr.dim - 1; i >= 0; --i) {
            if (++idx[i] < arr.shape[i]) {
                break;
            }
            idx[i] = 0;
            if (i == 0) {
                return min_val;
            }
        }
    }
}

template<typename T>
Array<T> sum(const Array<T>& arr, const std::vector<uint32_t>& axes = {}) {
    std::vector<uint32_t> idx(arr.dim, 0);
    Array<T> sum_val = T();
    while (true){
        uint32_t flat_index = arr.offset;
        for (size_t i = 0; i < arr.dim; ++i) {
            flat_index += idx[i] * arr.strides[i];
        }
        sum_val += (*arr.data_ptr)[flat_index];
        /* Increment the multi-dimensional index */
        for (int32_t i = arr.dim - 1; i >= 0; --i) {
            if (++idx[i] < arr.shape[i]) {
                break;
            }
            idx[i] = 0;
            if (i == 0) {
                return sum_val;
            }
        }
    }
}