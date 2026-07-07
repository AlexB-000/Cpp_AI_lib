#pragma once
#include "../deep_learning/module.hpp"

class PReLU: public Module{
public:
    unsigned int inputSize;
    unsigned int outputSize;
    Array<float> alpha;
    Array<float> inputCache;

    ~PReLU() = default;

    PReLU(unsigned int size, float inAlpha=0.01f) : inputSize(size), outputSize(size),
        alpha({1}, inAlpha), inputCache(std::vector<uint32_t>{size}) {}
    PReLU(unsigned int inInputSize, unsigned int inOutputSize, float inAlpha=0.01f) :
        inputSize(inInputSize), outputSize(inOutputSize),
        alpha({1}, inAlpha), inputCache(std::vector<uint32_t>{inInputSize}) {}

    std::shared_ptr<Module> copy() const override {
        return std::make_shared<PReLU>(*this);
    }

	std::vector< Array<float>* > get_parameters() override { return {&alpha}; };

    Array<float> forward(const Array<float>& input) override {
        if (input.dim != 1){
            throw std::invalid_argument("In PReLU forward : Input must be a 1D array.");
        }
        if (input.shape[0] != inputSize) {
            throw std::invalid_argument("In PReLU forward : Input size does not match the expected size.");
        }

        if(training) inputCache = input;
        
        Array<float> output {input.copy()};
        for (uint32_t i=0; i < output.shape[0]; i++){
            if (output.at(i) < 0){
                output.at(i) *= alpha.at(0);
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

        Array<float> deriv {prevDeriv.copy()};
        for (uint32_t i=0; i < deriv.shape[0]; i++){
            if (inputCache.at(i) <= 0){
                deriv.at(i) *= alpha.at(0);
            }
        }

        Array<float> d_alpha{{1}, 0.0f};
        for (uint32_t i=0; i < deriv.shape[0]; i++){
            if (inputCache.at(i) <= 0){
                d_alpha.at(0) += prevDeriv.at(i) * inputCache.at(i);
            }
        }
        return {d_alpha, deriv};
    }
};