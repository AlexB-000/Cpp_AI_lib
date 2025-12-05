#pragma once
#include "../Tensors/tensorMath.hpp"
#include "module.hpp"

class Linear: public Module{
public:
    int inputSize;
    int outputSize;
    Tensor<double> prevInput;
    Tensor<double> biases;
    Tensor<double> weights;

    Linear(int inInputSize, int inOutputSize) : inputSize(inInputSize), outputSize(inOutputSize) {
        biases = Tensor<double>(outputSize);
        weights = Tensor<double>({static_cast<unsigned int>(inputSize), static_cast<unsigned int>(outputSize)});
        prevInput = Tensor<double>(inputSize);
    }
	std::vector< Tensor<double>* > parameters() override;
    Tensor<double> forward(const Tensor<double>& input) override;
    std::vector< Tensor<double> > backward(const Tensor<double>& prevDeriv) override;
};