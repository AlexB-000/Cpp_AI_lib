#pragma once
#include <math.h>
#include "module.hpp"
#include "../utils.hpp"

class Softmax: public Module{
public:
    unsigned int inputSize;
    unsigned int outputSize;
    Array<float> inputCache;

    ~Softmax() = default;

    Softmax(unsigned int inInputSize, unsigned int inOutputSize) : inputSize(inInputSize), outputSize(inOutputSize) {}

    std::shared_ptr<Module> copy() const override {
        return std::make_shared<Softmax>(*this);
    }

	std::vector< Array<float>* > get_parameters() override { return {}; };

    Array<float> forward(const Array<float>& input) override {
        if (input.dim != 1){
            throw std::invalid_argument("In Softmax forward : Input must be a 1D array.");
        }
        if (input.shape[0] != inputSize) {
            throw std::invalid_argument("In Softmax forward : Input size does not match the expected size.");
        }

        if(training) inputCache = input;
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

        return output;
    }

    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override {
        if (prevDeriv.dim != 1){
            throw std::invalid_argument("In Softmax backward : Previous derivative must be a 1D array.");
        }
        if (prevDeriv.shape[0] != outputSize) {
            throw std::invalid_argument("In Softmax backward : Previous derivative size does not match the expected output size.");
        }

        Array<float> deriv{prevDeriv.shape};
        for (uint32_t i=0; i < deriv.shape[0]; i++){
            deriv[i] = prevDeriv[i]; // Placeholder, proper softmax derivative implementation needed
        }
        return {{}, deriv};
    }
};