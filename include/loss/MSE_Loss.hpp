#pragma once
#include "./Loss_base.hpp"

class MSELoss: public Loss{
public:
    ~MSELoss() = default;

    MSELoss(int inSize) : Loss(inSize) {}

	float compute(const Tensor& predicted, const Tensor& target) override {
        float loss = 0.0f;
        for (size_t i = 0; i < predicted.shape[0]; ++i) {
            float diff = predicted[i] - target[i];
            loss += diff * diff;
            gradient[i] = 2.0f * diff;
        }
        return loss / static_cast<float>(predicted.shape[0]);
    }
};