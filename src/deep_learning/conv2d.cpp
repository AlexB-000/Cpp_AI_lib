#include "CppAI/deep_learning/conv2d.hpp"

Array<float> Conv2d::forward(const Array<float>& input){
    if (input.shape != inputShape) {
        throw std::invalid_argument("In Conv2d forward : Input shape does not match the expected shape.");
    }
    if(training) inputCache = input;
    Array<float> output(outputShape, 0.0f);
    for (int32_t i=0; i<outputShape[0]; ++i){
        int32_t current_i = i*stride[0];
        for (int32_t j=0; j<outputShape[1]; ++j){
            int32_t current_j = j*stride[1];
            uint32_t outOffset = i * output.strides[0] + j*output.strides[1];

            for (int32_t m=0; m < filterSize[0]; ++m){
                int32_t input_i = current_i + m - filterSize[0] / 2;
                if (input_i < 0) {
                    if (padding == Padding::Replication) input_i = 0;
                    else if (padding == Padding::Reflection) input_i = -input_i;
                    else if (padding == Padding::Circular) input_i += inputShape[0];
                    else if (padding == Padding::Zero) continue;
                } else if (input_i >= inputShape[0]) {
                    if (padding == Padding::Replication) input_i = inputShape[0]-1;
                    else if (padding == Padding::Reflection) input_i = 2*inputShape[0]-input_i-2;
                    else if (padding == Padding::Circular) input_i -= inputShape[0];
                    else if (padding == Padding::Zero) continue;
                }
                for (int32_t n=0; n < filterSize[1]; ++n){
                    int32_t input_j = current_j + n - filterSize[1] / 2;
                    if (input_j < 0) {
                        if (padding == Padding::Replication) input_j = 0;
                        else if (padding == Padding::Reflection) input_j = -input_j;
                        else if (padding == Padding::Circular) input_j += inputShape[1];
                        else if (padding == Padding::Zero) continue;
                    } else if (input_j >= inputShape[1]) {
                        if (padding == Padding::Replication) input_j = inputShape[1]-1;
                        else if (padding == Padding::Reflection) input_j = 2*inputShape[1]-input_j-2;
                        else if (padding == Padding::Circular) input_j -= inputShape[1];
                        else if (padding == Padding::Zero) continue;
                    }
                    uint32_t inOffset = input.offset + input_i * input.strides[0] + input_j * input.strides[1];
                    uint32_t filterOffset = m * filters[0].strides[0] + n * filters[0].strides[1];
                    std::vector<uint32_t> indices (filterSize.size()-2, 0);
                    bool computing = true;
                    while (computing) {
                        uint32_t flatIdx = inOffset;
                        uint32_t filterIdx = filterOffset;
                        for (uint32_t k=0; k < filterSize.size()-2; ++k) {
                            flatIdx += indices[k] * input.strides[2+k];
                            filterIdx += indices[k] * filters[0].strides[2+k];
                        }
                        float value = (*input.data_ptr)[flatIdx];
                        for (uint32_t f=0; f < features; ++f) {
                            uint32_t outIdx = outOffset + f;
                            (*output.data_ptr)[outIdx] += value * (*filters[f].data_ptr)[filterIdx];
                        }
                        // Increment indices
                        for (uint32_t k=filterSize.size()-3; k >=0 ; --k) {
                            if (++indices[k] < filterSize[2+k]) {
                                break;
                            }
                            indices[k] = 0;
                            if (k == 0) {
                                computing = false;
                                break;
                            }
                        }
                    }
                }
            }            
        }
    }
    return output;
}

std::vector<Array<float>> Conv2d::backward(const Array<float>& prevDeriv){
    if (prevDeriv.shape != outputShape) {
        throw std::invalid_argument("In Conv2d backward : Previous derivative shape does not match the expected shape.");
    }
    std::vector<Array<float>> d_filters(features, Array<float>(filterSize, 0.0f));
    Array<float> deriv(inputShape, 0.0f);
    for (int32_t i=0; i<outputShape[0]; ++i){
        int32_t current_i = i*stride[0];
        for (int32_t j=0; j<outputShape[1]; ++j){
            int32_t current_j = j*stride[1];
            uint32_t outOffset = prevDeriv.offset + i * prevDeriv.strides[0] + j*prevDeriv.strides[1];

            for (int32_t m=0; m < filterSize[0]; ++m){
                int32_t input_i = current_i + m - filterSize[0] / 2;
                if (input_i < 0) {
                    if (padding == Padding::Replication) input_i = 0;
                    else if (padding == Padding::Reflection) input_i = -input_i;
                    else if (padding == Padding::Circular) input_i += inputShape[0];
                    else if (padding == Padding::Zero) continue;
                } else if (input_i >= inputShape[0]) {
                    if (padding == Padding::Replication) input_i = inputShape[0]-1;
                    else if (padding == Padding::Reflection) input_i = 2*inputShape[0]-input_i-2;
                    else if (padding == Padding::Circular) input_i -= inputShape[0];
                    else if (padding == Padding::Zero) continue;
                }
                for (int32_t n=0; n < filterSize[1]; ++n){
                    int32_t input_j = current_j + n - filterSize[1] / 2;
                    if (input_j < 0) {
                        if (padding == Padding::Replication) input_j = 0;
                        else if (padding == Padding::Reflection) input_j = -input_j;
                        else if (padding == Padding::Circular) input_j += inputShape[1];
                        else if (padding == Padding::Zero) continue;
                    } else if (input_j >= inputShape[1]) {
                        if (padding == Padding::Replication) input_j = inputShape[1]-1;
                        else if (padding == Padding::Reflection) input_j = 2*inputShape[1]-input_j-2;
                        else if (padding == Padding::Circular) input_j -= inputShape[1];
                        else if (padding == Padding::Zero) continue;
                    }
                    uint32_t inOffset = input_i * deriv.strides[0] + input_j * deriv.strides[1];
                    uint32_t filterOffset = m * filters[0].strides[0] + n * filters[0].strides[1];
                    std::vector<uint32_t> indices (filterSize.size()-2, 0);
                    bool computing = true;
                    while (computing) {
                        uint32_t inIdx = inOffset;
                        uint32_t filterIdx = filterOffset;
                        for (uint32_t k=0; k < filterSize.size()-2; ++k) {
                            inIdx += indices[k] * deriv.strides[2+k];
                            filterIdx += indices[k] * filters[0].strides[2+k];
                        }
                        for (uint32_t f=0; f < features; ++f) {
                            uint32_t outIdx = outOffset + f;
                            (*d_filters[f].data_ptr)[filterIdx] += (*prevDeriv.data_ptr)[outIdx] * (*inputCache.data_ptr)[inIdx];
                            (*deriv.data_ptr)[inIdx] += (*prevDeriv.data_ptr)[outIdx] * (*filters[f].data_ptr)[filterIdx];
                        }
                        // Increment indices
                        for (uint32_t k=filterSize.size()-3; k >=0 ; --k) {
                            if (++indices[k] < filterSize[2+k]) {
                                break;
                            }
                            indices[k] = 0;
                            if (k == 0) {
                                computing = false;
                                break;
                            }
                        }
                    }
                }
            }            
        }
    }
    d_filters.push_back(deriv);
    return d_filters;
}