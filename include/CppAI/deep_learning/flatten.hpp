#pragma once
#include "Arrays/ndarrays.hpp"
#include "module.hpp"

class Flatten: public Module {
public:
    std::vector<unsigned int> inputShape;
    unsigned int outputSize;

    ~Flatten() = default;

    Flatten(std::vector<unsigned int> inInputShape) :
        inputShape(inInputShape)
    {
        outputSize = 1;
        for (const auto& dim : inputShape) {
            outputSize *= dim;
        }
    }

    std::shared_ptr<Module> copy() const override {
        return std::make_shared<Flatten>(*this);
    }
	std::vector< Array<float>* > get_parameters() override { return {}; };

    Array<float> forward(const Array<float>& input) override {
        if (input.shape != inputShape) {
            throw std::invalid_argument("In Flatten forward : Input shape does not match the expected shape.");
        }
        if (!input.owner) {
            Array<float> inputCopy = input.copy();
            return Array<float> (std::vector<unsigned int>{outputSize}, *inputCopy.data_ptr);
        } else {
            return Array<float> (std::vector<unsigned int>{outputSize}, *input.data_ptr);
        }
    }
    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override {
        if (!prevDeriv.owner) {
            Array<float> prevDerivCopy = prevDeriv.copy();
            return {Array<float> (inputShape, *prevDerivCopy.data_ptr)};
        } else {
            return {Array<float> (inputShape, *prevDeriv.data_ptr)};
        }
    }
};
