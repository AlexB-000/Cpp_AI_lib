#pragma once
#include "./Loss.hpp"

class MSELoss: public Loss{
public:
    ~MSELoss() = default;

    MSELoss(int inSize) : Loss(inSize) {}

	float compute(const Array<float>& predicted, const Array<float>& target) override {
        if (predicted.shape != target.shape) {
            throw std::invalid_argument("In MSELoss : Predicted and target arrays must have the same shape.");
        }
        if (predicted.shape[0] != size) {
            throw std::invalid_argument("In MSELoss : Input size does not match the expected size.");
        }

        float loss = nd::sum((predicted - target) * (predicted - target)).at(0);

        gradient = (predicted - target) * 2.0f / static_cast<float>(size);
        
        return loss / static_cast<float>(size);
    }
};