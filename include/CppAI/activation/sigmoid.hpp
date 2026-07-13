#pragma once
#include <math.h>
#include "CppAI/deep_learning/module.hpp"

class Sigmoid: public Module{
public:
    bool skipTraining = false;
    unsigned int inputSize;
    unsigned int outputSize;
    Array<float> outputCache;

    ~Sigmoid() = default;
    Sigmoid(unsigned int size) : inputSize(size), outputSize(size), outputCache(std::vector<uint32_t>{size}) {}
    Sigmoid(unsigned int inInputSize, unsigned int inOutputSize) : inputSize(inInputSize), outputSize(inOutputSize),
        outputCache(std::vector<uint32_t>{inOutputSize}) {}

    void setSkipTraining(bool value=true) { skipTraining=value; }

    std::shared_ptr<Module> copy() const override {
        return std::make_shared<Sigmoid>(*this);
    }

	std::vector< Array<float>* > get_parameters() override { return {}; };

    Array<float> forward(const Array<float>& input) override {
        if (skipTraining && training) {
            return input;
        }
        
        if (input.dim != 1){
            throw std::invalid_argument("In Sigmoid forward : Input must be a 1D array.");
        }
        if (input.shape[0] != inputSize) {
            throw std::invalid_argument("In Sigmoid forward : Input size does not match the expected size.");
        }

        Array<float> output = 1.0f / (1.0f + std::exp( -(*input.data_ptr)[input.offset] ));

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
        return {(*prevDeriv.data_ptr)[0] * (*outputCache.data_ptr)[0] * (1.0f - (*outputCache.data_ptr)[0])}; // Sigmoid derivative
    }
};