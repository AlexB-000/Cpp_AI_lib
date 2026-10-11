#pragma once
#include "CppAI/deep_learning/module.hpp"
#include <math.h>

class Tanh: public Module{
public:
    Array<float> outputCache;

    ~Tanh() = default;

    Tanh(unsigned int size) : outputCache(std::vector<uint32_t>{size}) {}
    Tanh(unsigned int inInputSize, unsigned int inOutputSize)
    : outputCache(std::vector<uint32_t>{inOutputSize}) {}

    std::shared_ptr<Module> copy() const override {
        return std::make_shared<Tanh>(*this);
    }

	std::vector< Array<float>* > get_parameters() override { return {}; };

    Array<float> forward(const Array<float>& input) override {
        if (input.shape != outputCache.shape) {
            throw std::invalid_argument("In Tanh forward : Input shape does not match the expected shape.");
        }
        Array<float> output(input.copy());
        for (uint32_t i=0; i<output.data_ptr->size(); i++){
            (*output.data_ptr)[i] = std::tanh((*output.data_ptr)[i]);
        }
        
        if(training) outputCache = output;

        return output;
    }

    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override {
        if (prevDeriv.shape != outputCache.shape) {
            throw std::invalid_argument("In Tanh backward : Previous derivative shape does not match the expected shape.");
        }
        Array<float> deriv(prevDeriv.copy());
        for (uint32_t i=0; i<deriv.data_ptr->size(); i++){
            (*deriv.data_ptr)[i] = (*deriv.data_ptr)[i] * (1.0f - (*outputCache.data_ptr)[i] * (*outputCache.data_ptr)[i]);
        }
        return {deriv};
    }
};