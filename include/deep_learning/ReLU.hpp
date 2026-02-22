#pragma once
#include "module.hpp"

class ReLU: public Module{
public:
    unsigned int inputSize;
    unsigned int outputSize;
    Tensor inputCache;

    ~ReLU() = default;

    ReLU(unsigned int size) : inputSize(size), outputSize(size) {}
    ReLU(unsigned int inInputSize, unsigned int inOutputSize) : inputSize(inInputSize), outputSize(inOutputSize) {}

	std::vector< Tensor* > get_parameters() override { return {}; };

    Tensor forward(const Tensor& input){
        if (input.dim() != 1){
            throw std::invalid_argument("In ReLU forward : Input must be a 1D tensor.");
        }
        if (input.shape[0] != inputSize) {
            throw std::invalid_argument("In ReLU forward : Input size does not match the expected size.");
        }

        inputCache = input;
        Tensor output {input.shape};
        for (uint32_t i=0; i < output.shape[0]; i++){
            if (input[i] < 0){
                output[i] = 0;
            } else {
                output[i] = input[i];
            }
        }
        return output;
    }

    std::vector< Tensor > backward(const Tensor& prevDeriv) override {
        if (prevDeriv.dim() != 1){
            throw std::invalid_argument("In ReLU backward : Previous derivative must be a 1D tensor.");
        }
        if (prevDeriv.shape[0] != outputSize) {
            throw std::invalid_argument("In ReLU backward : Previous derivative size does not match the expected output size.");
        }

        Tensor deriv {prevDeriv.shape};
        for (uint32_t i=0; i < deriv.shape[0]; i++){
            if (inputCache[i] <= 0){
                deriv[i] = 0;
            } else {
                deriv[i] = prevDeriv[i];
            }
        }
        return {{}, deriv};
    }
};