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

        float loss = nd::sum(nd::abs(predicted - target)).at(0);

        for (uint32_t i = 0; i < predicted.shape[0]; ++i) {
            if (target.at(i) - predicted.at(i) < 0){
                gradient.at(i) = -1 / static_cast<float>(size);
            }
            else{
                gradient.at(i) = 1 / static_cast<float>(size);
            }
        }
        return loss / static_cast<float>(size);
    }
};