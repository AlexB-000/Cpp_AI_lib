#include "CppAI/deep_learning/pooling.hpp"

Array<float> Pool2dLayer::forward(const Array<float>& input){
    if (input.shape != inputShape) {
        throw std::invalid_argument("In Pool2dLayer forward : Input shape does not match the expected shape.");
    }
    Array<float> output(outputShape);
    for (uint32_t i=0; i<outputShape[0]; ++i){
        for (uint32_t j=0; j<outputShape[1]; ++j){
            unsigned int input_i = i * stride;
            unsigned int input_j = j * stride;

            // Pooling for this region

            std::vector<float> pool_values (outputShape[2], 0.0f);
            if (poolingType == Pooling::Max) {
                for (uint32_t k=0; k<outputShape[2]; ++k){
                    pool_values[k] = -std::numeric_limits<float>::infinity();
                }
                // Max pooling over each feature map
                for (unsigned int m=0; m<poolingSize[0]; ++m){
                    for (unsigned int n=0; n<poolingSize[1]; ++n){
                        unsigned int current_i = input_i + m;
                        unsigned int current_j = input_j + n;
                        if (current_i < inputShape[0] && current_j < inputShape[1]) {
                            for (uint32_t k=0; k<outputShape[2]; ++k){
                                unsigned int current_index = current_i*input.strides[0] + current_j*input.strides[1] + k*input.strides[2] + input.offset;
                                float current_value = (*input.data_ptr)[current_index];
                                if (current_value > pool_values[k]) {
                                    pool_values[k] = current_value;
                                }
                            }
                        }
                    }
                }
            } else if (poolingType == Pooling::Average) {
                unsigned int count = 0;
                // Sum over each feature map
                for (unsigned int m=0; m<poolingSize[0]; ++m){
                    for (unsigned int n=0; n<poolingSize[1]; ++n){
                        unsigned int current_i = input_i + m;
                        unsigned int current_j = input_j + n;
                        if (current_i < inputShape[0] && current_j < inputShape[1]) {
                            for (uint32_t k=0; k<outputShape[2]; ++k){
                                unsigned int current_index = current_i*input.strides[0] + current_j*input.strides[1] + k*input.strides[2] + input.offset;
                                pool_values[k] += (*input.data_ptr)[current_index];
                            }
                            count++;
                        }
                    }
                }
                // Average the values for each feature map
                for (uint32_t k=0; k<outputShape[2]; ++k){
                    pool_values[k] /= count;
                }
            }
            // store the results for this region
            for (uint32_t k=0; k<outputShape[2]; ++k){
                (*output.data_ptr)[i*output.strides[0] + j*output.strides[1] + k*output.strides[2] + output.offset] =
                    pool_values[k];
            }
        }
    }
    if(training){
        inputCache = input;
        if (poolingType == Pooling::Max)
            outputCache = output;
    }
    return output;
}

std::vector<Array<float>> Pool2dLayer::backward(const Array<float>& prevDeriv){
    if (prevDeriv.shape != outputShape) {
        throw std::invalid_argument("In Pool2dLayer backward : Previous derivative shape does not match the expected shape.");
    }
    Array<float> deriv(inputShape);
    for (uint32_t i=0; i<outputShape[0]; ++i){
        for (uint32_t j=0; j<outputShape[1]; ++j){
            unsigned int input_i = i * stride;
            unsigned int input_j = j * stride;

            // Pooling derivative for this region

            if (poolingType == Pooling::Max) {
                // Max pooling derivative over each feature map
                for (unsigned int m=0; m<poolingSize[0]; ++m){
                    for (unsigned int n=0; n<poolingSize[1]; ++n){
                        unsigned int current_i = input_i + m;
                        unsigned int current_j = input_j + n;
                        if (current_i < inputShape[0] && current_j < inputShape[1]) {

                            for (uint32_t k=0; k<outputShape[2]; ++k){
                                unsigned int current_index = current_i*inputCache.strides[0] + current_j*inputCache.strides[1] + k;
                                if ((*inputCache.data_ptr)[current_index] >= (*outputCache.data_ptr)[i*outputCache.strides[0] + j*outputCache.strides[1] + k]) {
                                    (*deriv.data_ptr)[current_index] += (*prevDeriv.data_ptr)[i*prevDeriv.strides[0] + j*prevDeriv.strides[1] + k*prevDeriv.strides[2] + prevDeriv.offset];
                                }
                            }
                        }
                    }
                }
            } else if (poolingType == Pooling::Average) {
                // Derivative over each feature map
                float norm = 1.0f / (std::min(poolingSize[0], inputShape[0] - input_i) *
                                     std::min(poolingSize[1], inputShape[1] - input_j));
                for (unsigned int m=0; m<poolingSize[0]; ++m){
                    for (unsigned int n=0; n<poolingSize[1]; ++n){
                        unsigned int current_i = input_i + m;
                        unsigned int current_j = input_j + n;
                        if (current_i < inputShape[0] && current_j < inputShape[1]) {
                            for (uint32_t k=0; k<outputShape[2]; ++k){
                                unsigned int current_index = current_i*inputCache.strides[0] + current_j*inputCache.strides[1] + k;
                                (*deriv.data_ptr)[current_index] += norm *
                                    (*prevDeriv.data_ptr)[i*prevDeriv.strides[0] + j*prevDeriv.strides[1] + k*prevDeriv.strides[2] + prevDeriv.offset];
                            }
                        }
                    }
                }
            }
        }
    }
    return {deriv};
}