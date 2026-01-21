#pragma once
#include "../../Tensor/include/tensorMath.hpp"

float max(const Tensor& tensor){
    float max_val = tensor[0];
    if (tensor.dim() == 0) {
        return max_val;
    }
    if (tensor.dim() > 1) {
        for (size_t i = 0; i < tensor.shape[0]; ++i) {
            Tensor subtensor = tensor.subtensor(i);
            float sub_max = max(subtensor);
            if (sub_max > max_val) {
                max_val = sub_max;
            }
        }
        return max_val;
    }

    for (size_t i = 1; i < tensor.shape[0]; ++i) {
        if (tensor[i] > max_val) {
            max_val = tensor[i];
        }
    }
    return max_val;
}