#pragma once
#include "module.hpp"
#include <math.h>

class Tanh: public Module{
public:
    unsigned int inputSize;
    unsigned int outputSize;
    Tensor inputCache;

    ~Tanh() = default;

    Tanh(unsigned int size) : inputSize(size), outputSize(size) {}
    Tanh(unsigned int inInputSize, unsigned int inOutputSize) : inputSize(inInputSize), outputSize(inOutputSize) {}

	std::vector< Tensor* > get_parameters() override { return {}; };

    Tensor forward(const Tensor& input){
        inputCache = input;
        Tensor output {input.shape};

        for (uint32_t i=0; i < output.shape[0]; i++){
            output[i] = std::tanh(input[i]);
        }

        return output;
    }

    std::vector< Tensor > backward(const Tensor& prevDeriv) override {
        Tensor deriv{prevDeriv.shape};
        for (uint32_t i=0; i < deriv.shape[0]; i++){
            float tanh_x = std::tanh(inputCache[i]);
            deriv[i] = prevDeriv[i] * (1 - tanh_x * tanh_x);
        }
        return {{}, deriv};
    }
};