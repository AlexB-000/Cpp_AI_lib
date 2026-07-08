#pragma once
#include <cmath>
#include "./Loss.hpp"

class CrossEntropyLoss: public Loss{
    float _epsilon = 1e-9f; // small value to prevent log(0)
public:
    ~CrossEntropyLoss() = default;

    CrossEntropyLoss(int size, float epsilon=1e-9f) : Loss(size), _epsilon(epsilon) {}

	float compute(const Array<float>& predicted, const Array<float>& target) override {
        if (predicted.shape != target.shape) {
            throw std::invalid_argument("In CrossEntropyLoss : Predicted and target arrays must have the same shape.");
        }
        if (predicted.shape[0] != size) {
            throw std::invalid_argument("In CrossEntropyLoss : Input size does not match the expected size.");
        }
        gradient = -target / (predicted + _epsilon);
        return -nd::sum(target * nd::log(predicted + _epsilon)).at(0);
    }
};