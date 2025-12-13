#pragma once
#include "../../Tensors/include/tensorMath.hpp"
#include "module.hpp"

class Network: public Module{
public:
    std::vector<std::shared_ptr<Module>> modules;
    unsigned int inputSize;
    unsigned int outputSize;
    //std::vector< Tensor<double> > layerInputs;
    std::vector<std::vector< Tensor<double> * >> parameters;
	
	Network(unsigned int inInputSize, unsigned int inOutputSize) : inputSize(inInputSize), outputSize(inOutputSize) {}
	
	void stackLayer(const Module& layer);

    Tensor<double> forward(const Tensor<double>& input) override;
    std::vector< Tensor<double> > backward(const Tensor<double>& prevDeriv) override;
};