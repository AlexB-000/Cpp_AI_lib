#pragma once
#include <cmath>
#include "Arrays/array.hpp"

namespace nd {
    //MARK: max
    template<typename T>
    Array<T> max(const Array<T>& arr, std::vector<uint32_t> axes = {}, bool keepdims = false) {
        if (axes.size() == 0) {
            axes.resize(arr.dim);
            for (uint32_t i = 0; i < arr.dim; ++i) {
                axes[i] = i;
            }
        }
        std::vector<uint32_t> new_shape(arr.dim);
        for (uint32_t i = 0; i < arr.dim; ++i) {
            if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                // this axis is not being reduced, so keep its size and stride
                new_shape[i] = arr.shape[i];
            } else if (keepdims) {
                new_shape[i] = 1;
            }
        }
        Array<T> result(new_shape);
        Array<bool> initialized(new_shape, false);

        std::vector<uint32_t> strides(arr.dim);
        uint32_t new_idx = 0;
        for (uint32_t i = 0; i < arr.dim; ++i) {
            if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                // this axis is not being reduced, so keep its size and stride
                strides[i] = result.strides[new_idx++];
            } else {
                strides[i] = 0;
                if (keepdims) {
                    new_idx++;
                }
            }
        }

        std::vector<uint32_t> idx(arr.dim, 0);
        while (true){
            uint32_t  resultIdx = 0, arrIdx = arr.offset;
            for (uint32_t i = 0; i < arr.dim; ++i) {
                resultIdx += idx[i] * strides[i];
                arrIdx += idx[i] * arr.strides[i];
            }
            if (!(*initialized.data_ptr)[resultIdx] || (*arr.data_ptr)[arrIdx] > (*result.data_ptr)[resultIdx]) {
                (*result.data_ptr)[resultIdx] = (*arr.data_ptr)[arrIdx];
                (*initialized.data_ptr)[resultIdx] = true;
            }
            /* Increment the multi-dimensional index */
            for (int32_t i = arr.dim - 1; i >= 0; --i) {
                if (++idx[i] < arr.shape[i]) {
                    break;
                }
                idx[i] = 0;
                if (i == 0) {
                    return result;
                }
            }
        }
    }

    //MARK: max
    template<typename T>
    Array<T> min(const Array<T>& arr, std::vector<uint32_t> axes = {}, bool keepdims = false) {
        if (axes.size() == 0) {
            axes.resize(arr.dim);
            for (uint32_t i = 0; i < arr.dim; ++i) {
                axes[i] = i;
            }
        }
        std::vector<uint32_t> new_shape(arr.dim);
        for (uint32_t i = 0; i < arr.dim; ++i) {
            if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                // this axis is not being reduced, so keep its size and stride
                new_shape[i] = arr.shape[i];
            } else if (keepdims) {
                new_shape[i] = 1;
            }
        }
        Array<T> result(new_shape);
        Array<bool> initialized(new_shape, false);

        std::vector<uint32_t> strides(arr.dim);
        uint32_t new_idx = 0;
        for (uint32_t i = 0; i < arr.dim; ++i) {
            if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                // this axis is not being reduced, so keep its size and stride
                strides[i] = result.strides[new_idx++];
            } else {
                strides[i] = 0;
                if (keepdims) {
                    new_idx++;
                }
            }
        }

        std::vector<uint32_t> idx(arr.dim, 0);
        while (true){
            uint32_t  resultIdx = 0, arrIdx = arr.offset;
            for (uint32_t i = 0; i < arr.dim; ++i) {
                resultIdx += idx[i] * strides[i];
                arrIdx += idx[i] * arr.strides[i];
            }
            if (!(*initialized.data_ptr)[resultIdx] || (*arr.data_ptr)[arrIdx] < (*result.data_ptr)[resultIdx]) {
                (*result.data_ptr)[resultIdx] = (*arr.data_ptr)[arrIdx];
                (*initialized.data_ptr)[resultIdx] = true;
            }
            /* Increment the multi-dimensional index */
            for (int32_t i = arr.dim - 1; i >= 0; --i) {
                if (++idx[i] < arr.shape[i]) {
                    break;
                }
                idx[i] = 0;
                if (i == 0) {
                    return result;
                }
            }
        }
    }

    //MARK: sum
    template<typename T>
    Array<T> sum(const Array<T>& arr, std::vector<uint32_t> axes = {}, bool keepdims = false) {
        if (axes.size() == 0) {
            axes.resize(arr.dim);
            for (uint32_t i = 0; i < arr.dim; ++i) {
                axes[i] = i;
            }
        }
        std::vector<uint32_t> new_shape(arr.dim);
        for (uint32_t i = 0; i < arr.dim; ++i) {
            if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                // this axis is not being reduced, so keep its size and stride
                new_shape[i] = arr.shape[i];
            } else if (keepdims) {
                new_shape[i] = 1;
            }
        }
        Array<T> result(new_shape);

        std::vector<uint32_t> strides(arr.dim);
        uint32_t new_idx = 0;
        for (uint32_t i = 0; i < arr.dim; ++i) {
            if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                // this axis is not being reduced, so keep its size and stride
                strides[i] = result.strides[new_idx++];
            } else {
                strides[i] = 0;
                if (keepdims) {
                    new_idx++;
                }
            }
        }

        std::vector<uint32_t> idx(arr.dim, 0);
        while (true){
            uint32_t  resultIdx = 0, arrIdx = arr.offset;
            for (uint32_t i = 0; i < arr.dim; ++i) {
                resultIdx += idx[i] * strides[i];
                arrIdx += idx[i] * arr.strides[i];
            }
            (*result.data_ptr)[resultIdx] += (*arr.data_ptr)[arrIdx];

            /* Increment the multi-dimensional index */
            for (int32_t i = arr.dim - 1; i >= 0; --i) {
                if (++idx[i] < arr.shape[i]) {
                    break;
                }
                idx[i] = 0;
                if (i == 0) {
                    return result;
                }
            }
        }
    }

    #define defineStdFuction(func) \
    template<typename T> \
    Array<T> func(const Array<T>& arr) { \
        Array<T> result(arr.shape); \
        std::vector<uint32_t> idx(arr.dim, 0); \
        while (true){ \
            uint32_t  resultIDX = 0, arrIdx = arr.offset; \
            for (uint32_t i = 0; i < arr.dim; ++i) { \
                resultIDX += idx[i] * result.strides[i]; \
                arrIdx += idx[i] * arr.strides[i]; \
            } \
            (*result.data_ptr)[resultIDX] = std::func((*arr.data_ptr)[arrIdx]); \
            /* Increment the multi-dimensional index */ \
            for (int32_t i = arr.dim - 1; i >= 0; --i) { \
                if (++idx[i] < arr.shape[i]) { \
                    break; \
                } \
                idx[i] = 0; \
                if (i == 0) { \
                    return result; \
                } \
            } \
        } \
    }
    defineStdFuction(abs)
    defineStdFuction(exp)
    defineStdFuction(log)
    defineStdFuction(tanh)
}