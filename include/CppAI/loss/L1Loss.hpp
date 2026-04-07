#pragma once
#include "./Loss.hpp"

class L1Loss: public Loss{
public:
    ~L1Loss() = default;

    L1Loss(int inSize) : Loss(inSize) {}

	float compute(const Array<float>& predicted, const Array<float>& target) override {
        if (predicted.shape != target.shape) {
            throw std::invalid_argument("In L1Loss : Predicted and target arrays must have the same shape.");
        }
        if (predicted.shape[0] != size) {
            throw std::invalid_argument("In L1Loss : Input size does not match the expected size.");
        }

        float loss = 0.0f;
        for (uint32_t i = 0; i < predicted.shape[0]; ++i) {
            loss += std::abs(predicted[i] - target[i]);
            if (target[i] - predicted[i] < 0){
                gradient[i] = -1 / static_cast<float>(target.shape[0]);
            }
            else{
                gradient[i] = 1 / static_cast<float>(target.shape[0]);
            }
        }
        return loss / static_cast<float>(target.shape[0]);
    }
};