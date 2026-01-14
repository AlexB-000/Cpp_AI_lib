#pragma once
#include "../../Tensor/include/tensorMath.hpp"
#include "module.hpp"

class Linear: public Module{
public:
    unsigned int inputSize;
    unsigned int outputSize;
    Tensor prevInput;
    Tensor biases;
    Tensor weights;

    ~Linear() = default;

    Linear(unsigned int inInputSize, unsigned int inOutputSize) : inputSize(inInputSize), outputSize(inOutputSize) {
        biases = Tensor({outputSize}, 0.0);
        weights = Tensor({outputSize, inputSize});
        prevInput = Tensor(inputSize);
    }
	std::vector< Tensor* > get_parameters() override;
    Tensor forward(const Tensor& input) override;
    std::vector< Tensor > backward(const Tensor& prevDeriv) override;
};