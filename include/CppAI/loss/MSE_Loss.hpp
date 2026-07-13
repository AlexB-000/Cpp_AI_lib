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
        float loss = 0;
        for (uint32_t i = 0; i < size; ++i) {
            const float l = (*predicted.data_ptr)[i*predicted.strides[0] + predicted.offset] - (*target.data_ptr)[i*target.strides[0] + target.offset];
            loss += l*l;
            (*gradient.data_ptr)[i] = l * 2.0f / static_cast<float>(size);
        }
        return loss / static_cast<float>(size);
    }
};