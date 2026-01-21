#pragma once
#include "./Loss.hpp"

class MSELoss: public Loss{
public:
    ~MSELoss() = default;

    MSELoss(int inSize) : Loss(inSize) {}

	float compute(const Tensor& predicted, const Tensor& target) override {
        float loss = 0.0f;
        for (size_t i = 0; i < predicted.shape[0]; ++i) {
            float diff = predicted[i] - target[i];
            loss += diff * diff;
            gradient[i] = 2.0f * diff / static_cast<float>(target.shape[0]);
        }
        return loss / static_cast<float>(target.shape[0]);
    }
};