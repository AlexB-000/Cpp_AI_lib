#pragma once
#include "Arrays/cpp_arrays.hpp"

template<typename T>
T max(const Array<T>& arr) {
    T max_val = (*arr.data_ptr)[0];
    for (uint32_t i = 1; i < arr.data_ptr->size(); ++i) {
        if ((*arr.data_ptr)[i] > max_val) {
            max_val = (*arr.data_ptr)[i];
        }
    }
    return max_val;
}