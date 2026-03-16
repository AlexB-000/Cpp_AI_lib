#pragma once
#include "module.hpp"
#include <math.h>

class Tanh: public Module{
public:
    unsigned int inputSize;
    unsigned int outputSize;
    Array<float> inputCache;

    ~Tanh() = default;

    Tanh(unsigned int size) : inputSize(size), outputSize(size) {}
    Tanh(unsigned int inInputSize, unsigned int inOutputSize) : inputSize(inInputSize), outputSize(inOutputSize) {}

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

        if(training) inputCache = input;
        Array<float> output {input.shape};

        for (uint32_t i=0; i < output.shape[0]; i++){
            output[i] = std::tanh(input[i]);
        }

        return output;
    }

    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override {
        if (prevDeriv.dim != 1){
            throw std::invalid_argument("In Tanh backward : Previous derivative must be a 1D tensor.");
        }
        if (prevDeriv.shape[0] != outputSize) {
            throw std::invalid_argument("In Tanh backward : Previous derivative size does not match the expected output size.");
        }
        
        Array<float> deriv{prevDeriv.shape};
        for (uint32_t i=0; i < deriv.shape[0]; i++){
            float tanh_x = std::tanh(inputCache[i]);
            deriv[i] = prevDeriv[i] * (1 - tanh_x * tanh_x);
        }
        return {{}, deriv};
    }
};