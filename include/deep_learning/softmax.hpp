#pragma once
#include "module.hpp"
#include "../utils.hpp"
#include <math.h>

class Softmax: public Module{
public:
    unsigned int inputSize;
    unsigned int outputSize;
    Tensor inputCache;

    ~Softmax() = default;

    Softmax(unsigned int inInputSize, unsigned int inOutputSize) : inputSize(inInputSize), outputSize(inOutputSize) {}

	std::vector< Tensor* > get_parameters() override { return {}; };

    Tensor forward(const Tensor& input){
        if (input.dim() != 1){
            throw std::invalid_argument("In Softmax forward : Input must be a 1D tensor.");
        }
        if (input.shape[0] != inputSize) {
            throw std::invalid_argument("In Softmax forward : Input size does not match the expected size.");
        }

        inputCache = input;
        Tensor output {input.shape};

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

    std::vector< Tensor > backward(const Tensor& prevDeriv) override {
        if (prevDeriv.dim() != 1){
            throw std::invalid_argument("In Softmax backward : Previous derivative must be a 1D tensor.");
        }
        if (prevDeriv.shape[0] != outputSize) {
            throw std::invalid_argument("In Softmax backward : Previous derivative size does not match the expected output size.");
        }

        Tensor deriv{prevDeriv.shape};
        for (uint32_t i=0; i < deriv.shape[0]; i++){
            deriv[i] = prevDeriv[i]; // Placeholder, proper softmax derivative implementation needed
        }
        return {{}, deriv};
    }
};