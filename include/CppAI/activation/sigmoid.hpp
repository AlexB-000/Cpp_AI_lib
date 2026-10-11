#pragma once
#include <math.h>
#include "CppAI/deep_learning/module.hpp"

class Sigmoid: public Module{
public:
    bool skipTraining = false;
    Array<float> outputCache;

    ~Sigmoid() = default;
    Sigmoid(unsigned int size) : outputCache(std::vector<uint32_t>{size}) {}
    Sigmoid(std::vector<unsigned int> outputShape) : outputCache(outputShape) {}

    void setSkipTraining(bool value=true) { skipTraining=value; }

    std::shared_ptr<Module> copy() const override {
        return std::make_shared<Sigmoid>(*this);
    }

	std::vector< Array<float>* > get_parameters() override { return {}; };

    Array<float> forward(const Array<float>& input) override {
        if (skipTraining && training) {
            return input;
        }
        
        if (input.shape != outputCache.shape) {
            throw std::invalid_argument("In Sigmoid forward : Input shape does not match the expected shape.");
        }
        // Sigmoid
        Array<float> output(outputCache.shape);
        
        if (outputCache.shape.size() == 1 && outputCache.shape[0] == 1) {
            float sig = 1.0f / (1.0f + std::exp( -(*input.data_ptr)[input.offset] ));
            (*output.data_ptr)[0] = sig;
            (*outputCache.data_ptr)[outputCache.offset] = sig;
        } else {
            if (input.owner){
                for (uint32_t i=0; i<outputCache.data_ptr->size(); i++){
                    float sig = 1.0f / (1.0f + std::exp( -(*input.data_ptr)[i] ));
                    (*output.data_ptr)[i] = sig;
                }
            } else {
                std::vector<uint32_t> indices(input.dim, 0);
                for (uint32_t i=0; i<outputCache.data_ptr->size(); i++){
                    uint32_t inputIndex = input.offset;
                    for (uint32_t d=0; d<input.dim; d++){
                        inputIndex += indices[d] * input.strides[d];
                    }
                    float sig = 1.0f / (1.0f + std::exp( -(*input.data_ptr)[inputIndex] ));
                    (*output.data_ptr)[i] = sig;
                    // increment indices
                    for (int32_t d=input.dim-1; d>=0; d--){
                        indices[d]++;
                        if (indices[d] < input.shape[d]){
                            break;
                        } else {
                            indices[d] = 0;
                        }
                    }
                }
            }
            if (training) outputCache = output;
        }
        return output;
    }

    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override {
        if (skipTraining && training) {
            return {prevDeriv};
        }

        if (prevDeriv.shape != outputCache.shape) {
            throw std::invalid_argument("In Sigmoid backward : Previous derivative shape does not match the expected shape.");
        }
        Array<float> deriv(outputCache.shape);

        // fast path for 1 neurone output
        if (outputCache.shape.size() == 1 && outputCache.shape[0] == 1) {
            // Sigmoid derivative
            float dL = (*prevDeriv.data_ptr)[prevDeriv.offset] * (*outputCache.data_ptr)[0] * (1.0f - (*outputCache.data_ptr)[0]);
            (*deriv.data_ptr)[0] = dL;
            return {deriv};
        }

        // Sigmoid derivative
        for (uint32_t i=0; i<outputCache.data_ptr->size(); i++){
            float dy = (*outputCache.data_ptr)[i] * (1.0f - (*outputCache.data_ptr)[i]);
            (*deriv.data_ptr)[i] = dy;
        }
        if (prevDeriv.owner){
            for (uint32_t i=0; i<outputCache.data_ptr->size(); i++){
                (*deriv.data_ptr)[i] *= (*prevDeriv.data_ptr)[i];
            }
        } else {
            std::vector<uint32_t> indices(prevDeriv.dim, 0);
            for (uint32_t i=0; i<outputCache.data_ptr->size(); i++){
                uint32_t prevDerivIndex = prevDeriv.offset;
                for (uint32_t d=0; d<prevDeriv.dim; d++){
                    prevDerivIndex += indices[d] * prevDeriv.strides[d];
                }
                (*deriv.data_ptr)[i] *= (*prevDeriv.data_ptr)[prevDerivIndex];
                // increment indices
                for (int32_t d=prevDeriv.dim-1; d>=0; d--){
                    indices[d]++;
                    if (indices[d] < prevDeriv.shape[d]){
                        break;
                    } else {
                        indices[d] = 0;
                    }
                }
            }
        }
        return {deriv};
    }
};