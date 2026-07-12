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
        // BCEWithLogitsLoss formula
        float loss = std::log(1.0f + std::exp((*predicted.data_ptr)[predicted.offset])) -
                     (*target.data_ptr)[target.offset] * (*predicted.data_ptr)[predicted.offset];
        // Sigmoid(predicted) - target
        (*gradient.data_ptr)[0] = 1.0f / (1.0f + std::exp(-(*predicted.data_ptr)[predicted.offset])) -
                                (*target.data_ptr)[target.offset];
        return loss;
    }
};