#pragma once
#include "../../Tensor/include/tensorMath.hpp"
#include "module.hpp"

class Linear: public Module{
public:
    int inputSize;
    int outputSize;
    Tensor prevInput;
    Tensor biases;
    Tensor weights;

    ~Linear() = default;

    Linear(int inInputSize, int inOutputSize) : inputSize(inInputSize), outputSize(inOutputSize) {
        biases = Tensor(outputSize);
        weights = Tensor({static_cast<unsigned int>(outputSize), static_cast<unsigned int>(inputSize)});
        prevInput = Tensor(inputSize);
    }
	std::vector< Tensor* > get_parameters() override;
    Tensor forward(const Tensor& input) override;
    std::vector< Tensor > backward(const Tensor& prevDeriv) override;
};