#pragma once
#include "Tensors/tensorMath.hpp"

class Module{
public:
	virtual std::vector< Tensor<double>* > parameters();
    virtual Tensor<double> forward(const Tensor<double>& input) = 0;
    virtual std::vector< Tensor<double> > backward(const Tensor<double>& prevDeriv) = 0;
};