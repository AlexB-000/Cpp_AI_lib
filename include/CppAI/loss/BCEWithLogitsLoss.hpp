#pragma once
#include <cmath>
#include "./Loss.hpp"

class BCEWithLogitsLoss: public Loss{
public:
    ~BCEWithLogitsLoss() = default;

    BCEWithLogitsLoss() : Loss(1) {}

	float compute(const Array<float>& predicted, const Array<float>& target) override {
        if (predicted.shape != target.shape) {
            throw std::invalid_argument("In BCEWithLogitsLoss : Predicted and target arrays must have the same shape.");
        }
        if (predicted.shape[0] != size) {
            throw std::invalid_argument("In BCEWithLogitsLoss : Input size does not match the expected size.");
        }

        float loss = std::log(1 + std::exp(predicted.at(0))) - target.at(0) * predicted.at(0); // BCEWithLogitsLoss formula
        gradient.at(0) = 1.0f / (1.0f + std::exp(-predicted.at(0))) - target.at(0); // Sigmoid(predicted) - target
        return loss;
    }
};