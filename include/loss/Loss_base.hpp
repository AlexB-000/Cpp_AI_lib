#pragma once
#include <vector>

class Loss {
protected:
    int size;
    Tensor gradient;
public:
    virtual ~Loss() = default;
    Loss(unsigned int inSize) : size(inSize), gradient({inSize}, 1.0f) {}
    virtual float compute(const Tensor& predicted, const Tensor& target) = 0;
    virtual Tensor get_gradient() {return gradient;}
};