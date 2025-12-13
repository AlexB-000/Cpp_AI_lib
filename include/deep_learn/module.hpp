#pragma once
#include "../../Tensor/include/tensorMath.hpp"

class Module{
public:
	virtual std::vector< Tensor* > get_parameters() = 0;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual std::vector< Tensor > backward(const Tensor& prevDeriv) = 0;
};