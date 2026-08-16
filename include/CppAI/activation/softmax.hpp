#pragma once
#include <math.h>
#include "CppAI/deep_learning/module.hpp"

class Softmax: public Module{
public:
    bool skipTraining = false;
    unsigned int inputSize;
    unsigned int outputSize;
    Array<float> outputCache;

    ~Softmax() = default;
    Softmax(unsigned int size) : inputSize(size), outputSize(size), outputCache(std::vector<uint32_t>{size}) {}
    Softmax(unsigned int inInputSize, unsigned int inOutputSize) : inputSize(inInputSize), outputSize(inOutputSize),
        outputCache(std::vector<uint32_t>{inOutputSize}) {}

    void setSkipTraining(bool value=true) { skipTraining=value; }

    std::shared_ptr<Module> copy() const override {
        return std::make_shared<Softmax>(*this);
    }

	std::vector< Array<float>* > get_parameters() override { return {}; };

    Array<float> forward(const Array<float>& input) override {
        if (skipTraining && training) {
            return input;
        }
        
        if (input.dim != 1){
            throw std::invalid_argument("In Softmax forward : Input must be a 1D array.");
        }
        if (input.shape[0] != inputSize) {
            throw std::invalid_argument("In Softmax forward : Input size does not match the expected size.");
        }

        // find max
        float max_input = (*input.data_ptr)[input.offset];
        for (uint32_t i=0; i<input.shape[0]; i++){
            if ( (*input.data_ptr)[i * input.strides[0] + input.offset] > max_input )
                max_input = (*input.data_ptr)[i * input.strides[0] + input.offset];
        }

        // exp(logit - max)
        float sum = 0.0f;
        Array<float> output(input.shape);
        for (uint32_t i=0; i<output.shape[0]; i++){
            (*output.data_ptr)[i] = std::exp((*input.data_ptr)[i * input.strides[0] + input.offset] - max_input);
            sum += (*output.data_ptr)[i];
        }

        // normalisation
        for (uint32_t i=0; i<output.shape[0]; i++){
            (*output.data_ptr)[i] /= sum;
        }
        
        if (training) outputCache = output;
        return output;
    }

    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override {
        if (skipTraining && training) {
            return {prevDeriv};
        }
        if (prevDeriv.dim != 1){
            throw std::invalid_argument("In Softmax backward : Previous derivative must be a 1D array.");
        }
        if (prevDeriv.shape[0] != outputSize) {
            throw std::invalid_argument("In Softmax backward : Previous derivative size does not match the expected output size.");
        }
        // Softmax derivative
        Array<float> deriv(prevDeriv.shape, 0);
        for (uint32_t i=0; i<deriv.shape[0]; i++){
            for (uint32_t k=0; k<i; k++){
                // i != k
                (*deriv.data_ptr)[k] -= (*outputCache.data_ptr)[i] * (*outputCache.data_ptr)[k] *
                    (*prevDeriv.data_ptr)[i * prevDeriv.strides[0] + prevDeriv.offset];
            }
            // i = k
            (*deriv.data_ptr)[i] += (*outputCache.data_ptr)[i] * (1.0f - (*outputCache.data_ptr)[i]) * 
                    (*prevDeriv.data_ptr)[i * prevDeriv.strides[0] + prevDeriv.offset];

            for (uint32_t k=i+1; k<deriv.shape[0]; k++){
                // i != k
                (*deriv.data_ptr)[k] -= (*outputCache.data_ptr)[i] * (*outputCache.data_ptr)[k] *
                    (*prevDeriv.data_ptr)[i * prevDeriv.strides[0] + prevDeriv.offset];
            }
        }
        return {deriv};
    }
};