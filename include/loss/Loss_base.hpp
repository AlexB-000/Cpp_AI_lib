#pragma once
#include <vector>

class Loss {
protected:
    int size;
    std::vector<float> gradient;
public:
    virtual ~Loss() = default;
    Loss(int inSize) : size(inSize), gradient(inSize, 1.0f) {}
    virtual float compute(const std::vector<float>& predicted, const std::vector<float>& target) = 0;
    virtual std::vector<float> get_gradient() {return gradient;}
};