#pragma once
#include <math.h>
#include "CppAI/deep_learning/module.hpp"
#include "../utils.hpp"

class Softmax: public Module{
public:
    bool skipTraining = false;
    unsigned int inputSize;
    unsigned int outputSize;
    Array<float> outputCache;

    ~Softmax() = default;
    Softmax(unsigned int size) : inputSize(size), outputSize(size) {}
    Softmax(unsigned int inInputSize, unsigned int inOutputSize) : inputSize(inInputSize), outputSize(inOutputSize) {}

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

        Array<float> output {input.shape};

        // find max
        double max_input = max(input);

        // exp(logit - max)
        double sum = 0.0;
        for (unsigned int i = 0; i < input.shape[0]; ++i) {
            output[i] = std::exp(input[i] - max_input);
            sum += output[i];
        }

        // normalisation
        for (unsigned int i = 0; i < input.shape[0]; ++i) {
            output[i] /= sum;
        }
        if (training) outputCache = output;
        return output;
    }

    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override {
        if (skipTraining && training) {
            return {{}, prevDeriv};
        }

        if (prevDeriv.dim != 1){
            throw std::invalid_argument("In Softmax backward : Previous derivative must be a 1D array.");
        }
        if (prevDeriv.shape[0] != outputSize) {
            throw std::invalid_argument("In Softmax backward : Previous derivative size does not match the expected output size.");
        }

        Array<float> deriv{prevDeriv.shape, 0.0f};
        for (uint32_t k=0; k < deriv.shape[0]; k++){
            for (uint32_t i=0; i < deriv.shape[0]; i++){
                bool delta = 0.0f;
                if (i == k) delta = 1.0f;
                deriv[k] += prevDeriv[i] * outputCache[i] * (delta - outputCache[k]);
            }
        }
        return {{}, deriv};
    }
};