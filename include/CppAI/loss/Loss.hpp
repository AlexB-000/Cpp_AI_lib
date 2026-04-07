#pragma once
#include <vector>
#include "Arrays/cpp_arrays.hpp"

class Loss {
protected:
    int size;
    Array<float> gradient;
public:
    virtual ~Loss() = default;
    Loss(unsigned int inSize) : size(inSize), gradient({inSize}, 1.0f) {}
    virtual float compute(const Array<float>& predicted, const Array<float>& target) = 0;
    virtual Array<float> get_gradient() {return gradient;}
};