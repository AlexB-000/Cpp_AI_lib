#pragma once
#include <cmath>
#include "./Loss.hpp"

class CrossEntropyLoss: public Loss{
    float _epsilon = 1e-9f; // small value to prevent log(0)
public:
    ~CrossEntropyLoss() = default;

    CrossEntropyLoss(float epsilon=1e-9f) : Loss(1), _epsilon(epsilon) {}

	float compute(const Array<float>& predicted, const Array<float>& target) override {
        if (predicted.shape != target.shape) {
            throw std::invalid_argument("In CrossEntropyLoss : Predicted and target arrays must have the same shape.");
        }
        if (predicted.shape[0] != size) {
            throw std::invalid_argument("In CrossEntropyLoss : Input size does not match the expected size.");
        }

        float sum = 0.0f;
        for (uint32_t i = 0; i < predicted.shape[0]; ++i) {
            float loss = -target[i] * std::log(predicted[i] + _epsilon);
            sum += loss;
            gradient[i] = -target[i] / (predicted[i] + _epsilon);
        }
        return sum / static_cast<float>(target.shape[0]);
    }
};