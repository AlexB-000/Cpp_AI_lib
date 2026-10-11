#pragma once
#include "../deep_learning/module.hpp"

class PReLU: public Module{
public:
    Array<float> alpha;
    Array<float> inputCache;

    ~PReLU() = default;

    PReLU(unsigned int size, float inAlpha=0.01f)
        : alpha({1}, inAlpha), inputCache(std::vector<uint32_t>{size}) {}
    PReLU(std::vector<unsigned int> inputShape, float inAlpha=0.01f)
        : alpha({1}, inAlpha), inputCache(inputShape) {}

    std::shared_ptr<Module> copy() const override {
        return std::make_shared<PReLU>(*this);
    }

	std::vector< Array<float>* > get_parameters() override { return {&alpha}; };

    Array<float> forward(const Array<float>& input) override {
        if (input.shape != inputCache.shape) {
            throw std::invalid_argument("In PReLU forward : Input shape does not match the expected shape.");
        }
        
        if(training) inputCache = input;
        
        Array<float> output {input.copy()};
        for (uint32_t i=0; i < output.data_ptr->size(); i++){
            if ((*output.data_ptr)[i] < 0){
                (*output.data_ptr)[i] *= (*alpha.data_ptr)[0];
            }
        }
        return output;
    }

    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override {
        if (prevDeriv.shape != inputCache.shape) {
            throw std::invalid_argument("In PReLU backward : Previous derivative shape does not match the expected shape.");
        }

        Array<float> deriv {prevDeriv.copy()};

        Array<float> d_alpha{{1}, 0.0f};
        for (uint32_t i=0; i < deriv.data_ptr->size(); i++){
            if ((*inputCache.data_ptr)[i] <= 0){
                (*d_alpha.data_ptr)[0] += (*deriv.data_ptr)[i] * (*inputCache.data_ptr)[i];
            }
        }

        for (uint32_t i=0; i < deriv.data_ptr->size(); i++){
            if ((*inputCache.data_ptr)[i] <= 0){
                (*deriv.data_ptr)[i] *= (*alpha.data_ptr)[0];
            }
        }

        return {d_alpha, deriv};
    }
};