#pragma once
#include "../deep_learning/module.hpp"

class LeakyReLU: public Module{
public:
    float alpha; // slope for negative inputs

    Array<float> inputCache;

    ~LeakyReLU() = default;

    LeakyReLU(unsigned int size, float inAlpha=0.01f)
        : alpha(inAlpha), inputCache(std::vector<uint32_t>{size}) {}
    LeakyReLU(std::vector<unsigned int> inputShape, float inAlpha=0.01f)
        : alpha(inAlpha), inputCache(inputShape)  {}

    std::shared_ptr<Module> copy() const override {
        return std::make_shared<LeakyReLU>(*this);
    }

	std::vector< Array<float>* > get_parameters() override { return {}; };

    Array<float> forward(const Array<float>& input) override {
        if (input.shape != inputCache.shape) {
            throw std::invalid_argument("In LeakyReLU forward : Input shape does not match the expected shape.");
        }

        if(training) inputCache = input;
        Array<float> output {input.copy()};
        for (uint32_t i=0; i < output.data_ptr->size(); i++){
            if ((*output.data_ptr)[i] < 0){
                (*output.data_ptr)[i] *= alpha;
            }
        }
        return output;
    }

    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override {
        if (prevDeriv.shape != inputCache.shape) {
            throw std::invalid_argument("In LeakyReLU backward : Previous derivative shape does not match the expected shape.");
        }

        Array<float> deriv {prevDeriv.copy()};
        for (uint32_t i=0; i < deriv.data_ptr->size(); i++){
            if ((*inputCache.data_ptr)[i] <= 0){
                (*deriv.data_ptr)[i] *= alpha;
            }
        }
        return {deriv};
    }
};