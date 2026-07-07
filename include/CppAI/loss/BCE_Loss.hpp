#pragma once
#include <cmath>
#include "./Loss.hpp"

class BCELoss: public Loss{
    float _epsilon = 1e-9f; // small value to prevent log(0)
public:
    ~BCELoss() = default;

    BCELoss(float epsilon=1e-9f) : Loss(1), _epsilon(epsilon) {}

	float compute(const Array<float>& predicted, const Array<float>& target) override {
        if (predicted.shape != target.shape) {
            throw std::invalid_argument("In BCELoss : Predicted and target arrays must have the same shape.");
        }
        if (predicted.shape[0] != size) {
            throw std::invalid_argument("In BCELoss : Input size does not match the expected size.");
        }

        float loss = -(target.at(0) * std::log(predicted.at(0) + _epsilon) +
                       (1.0f - target.at(0)) * std::log(1.0f - predicted.at(0) + _epsilon));
        gradient.at(0) = ((1.0f - target.at(0)) / (1.0f - predicted.at(0) + _epsilon)) - (target.at(0) / (predicted.at(0) + _epsilon));
        return loss;
    }
};