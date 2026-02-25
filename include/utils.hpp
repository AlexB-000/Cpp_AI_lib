#pragma once
#include "../Arrays/cpp_arrays.hpp"

template<typename T>
T max(const Array<T>& arr) {
    T max_val = arr.data[0];
    for (uint32_t i = 1; i < arr.data.size(); ++i) {
        if (arr.data[i] > max_val) {
            max_val = arr.data[i];
        }
    }
    return max_val;
}