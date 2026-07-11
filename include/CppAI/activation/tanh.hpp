#pragma once
#include "CppAI/deep_learning/module.hpp"
#include <math.h>

class Tanh: public Module{
public:
    unsigned int inputSize;
    unsigned int outputSize;
    Array<float> outputCache;

    ~Tanh() = default;

    Tanh(unsigned int size) : inputSize(size), outputSize(size), outputCache(std::vector<uint32_t>{size}) {}
    Tanh(unsigned int inInputSize, unsigned int inOutputSize) : inputSize(inInputSize), outputSize(inOutputSize),
        outputCache(std::vector<uint32_t>{inOutputSize}) {}

    std::shared_ptr<Module> copy() const override {
        return std::make_shared<Tanh>(*this);
    }

	std::vector< Array<float>* > get_parameters() override { return {}; };

    Array<float> forward(const Array<float>& input) override {
        if (input.dim != 1){
            throw std::invalid_argument("In Tanh forward : Input must be a 1D tensor.");
        }
        if (input.shape[0] != inputSize) {
            throw std::invalid_argument("In Tanh forward : Input size does not match the expected size.");
        }
        Array<float> output = nd::tanh(input);
        
        if(training) outputCache = output;

        return output;
    }

    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override {
        if (prevDeriv.dim != 1){
            throw std::invalid_argument("In Tanh backward : Previous derivative must be a 1D tensor.");
        }
        if (prevDeriv.shape[0] != outputSize) {
            throw std::invalid_argument("In Tanh backward : Previous derivative size does not match the expected output size.");
        }
        return {prevDeriv * (Array<float>{1.0f} - outputCache * outputCache)};
    }
};