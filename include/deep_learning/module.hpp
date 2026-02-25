#pragma once
#include "../../Arrays/cpp_arrays.hpp"

class Module{
public:
	virtual std::vector< Array<float>* > get_parameters() = 0;
    virtual Array<float> forward(const Array<float>& input) = 0;
    virtual std::vector< Array<float> > backward(const Array<float>& prevDeriv) = 0;
};