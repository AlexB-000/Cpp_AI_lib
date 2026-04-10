#pragma once
#include "../deep_learning/module.hpp"

class ReLU: public Module{
public:
    unsigned int inputSize;
    unsigned int outputSize;
    Array<float> inputCache;

    ~ReLU() = default;

    ReLU(unsigned int size) : inputSize(size), outputSize(size) {}
    ReLU(unsigned int inInputSize, unsigned int inOutputSize) : inputSize(inInputSize), outputSize(inOutputSize) {}

    std::shared_ptr<Module> copy() const override {
        return std::make_shared<ReLU>(*this);
    }

	std::vector< Array<float>* > get_parameters() override { return {}; };

    Array<float> forward(const Array<float>& input) override {
        if (input.dim != 1){
            throw std::invalid_argument("In ReLU forward : Input must be a 1D array.");
        }
        if (input.shape[0] != inputSize) {
            throw std::invalid_argument("In ReLU forward : Input size does not match the expected size.");
        }

        if(training) inputCache = input;
        Array<float> output {input.shape};
        for (uint32_t i=0; i < output.shape[0]; i++){
            if (input[i] < 0){
                output[i] = 0;
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
                deriv[i] = 0;
            } else {
                deriv[i] = prevDeriv[i];
            }
        }
        return {{}, deriv};
    }
};