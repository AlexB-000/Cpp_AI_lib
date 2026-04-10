#pragma once
#include "../deep_learning/module.hpp"

class LeakyReLU: public Module{
public:
    unsigned int inputSize;
    unsigned int outputSize;
    float alpha; // slope for negative inputs

    Array<float> inputCache;

    ~LeakyReLU() = default;

    LeakyReLU(unsigned int size, float inAlpha=0.01f) : inputSize(size), outputSize(size), alpha(inAlpha) {}
    LeakyReLU(unsigned int inInputSize, unsigned int inOutputSize, float inAlpha=0.01f) :
        inputSize(inInputSize), outputSize(inOutputSize), alpha(inAlpha) {}

    std::shared_ptr<Module> copy() const override {
        return std::make_shared<LeakyReLU>(*this);
    }

	std::vector< Array<float>* > get_parameters() override { return {}; };

    Array<float> forward(const Array<float>& input) override {
        if (input.dim != 1){
            throw std::invalid_argument("In LeakyReLU forward : Input must be a 1D array.");
        }
        if (input.shape[0] != inputSize) {
            throw std::invalid_argument("In LeakyReLU forward : Input size does not match the expected size.");
        }

        if(training) inputCache = input;
        Array<float> output {input.shape};
        for (uint32_t i=0; i < output.shape[0]; i++){
            if (input[i] < 0){
                output[i] = alpha * input[i];
            } else {
                output[i] = input[i];
            }
        }
        return output;
    }

    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override {
        if (prevDeriv.dim != 1){
            throw std::invalid_argument("In ReLU backward : Previous derivative must be a 1D array.");
        }
        if (prevDeriv.shape[0] != outputSize) {
            throw std::invalid_argument("In ReLU backward : Previous derivative size does not match the expected output size.");
        }

        Array<float> deriv {prevDeriv.shape};
        for (uint32_t i=0; i < deriv.shape[0]; i++){
            if (inputCache[i] <= 0){
                deriv[i] = alpha * prevDeriv[i];
            } else {
                deriv[i] = prevDeriv[i];
            }
        }
        return {{}, deriv};
    }
};