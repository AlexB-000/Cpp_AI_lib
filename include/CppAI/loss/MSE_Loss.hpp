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

        float loss = 0.0f;
        for (uint32_t i = 0; i < predicted.shape[0]; ++i) {
            float diff = predicted.at(i) - target.at(i);
            loss += diff * diff;
            gradient.at(i) = 2.0f * diff / static_cast<float>(target.shape[0]);
        }
        return loss / static_cast<float>(target.shape[0]);
    }
};