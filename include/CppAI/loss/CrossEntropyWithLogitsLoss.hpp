#pragma once
#include <cmath>
#include "./Loss.hpp"

class CrossEntropyWithLogitsLoss: public Loss{
public:
    ~CrossEntropyWithLogitsLoss() = default;

    CrossEntropyWithLogitsLoss(int size) : Loss(size) {}

	float compute(const Array<float>& predicted, const Array<float>& target) override {
        if (predicted.shape != target.shape) {
            throw std::invalid_argument("In CrossEntropyWithLogitsLoss : Predicted and target arrays must have the same shape.");
        }
        if (predicted.shape[0] != size) {
            throw std::invalid_argument("In CrossEntropyWithLogitsLoss : Input size does not match the expected size.");
        }
        // stable softmax: subtract max logit
        float max_logit = (*predicted.data_ptr)[predicted.offset];
        for (uint32_t i = 0; i < predicted.shape[0]; ++i) {
            float v = (*predicted.data_ptr)[i * predicted.strides[0] + predicted.offset];
            if (v > max_logit) max_logit = v;
        }

        float sum_exp = 0.0f;
        std::vector<float> exps(predicted.shape[0]);
        for (uint32_t i = 0; i < predicted.shape[0]; ++i) {
            float v = std::exp((*predicted.data_ptr)[i * predicted.strides[0] + predicted.offset] - max_logit);
            exps[i] = v;
            sum_exp += v;
        }

        float loss = 0.0f;
        for (uint32_t i = 0; i < gradient.shape[0]; ++i) {
            float p = exps[i] / sum_exp;
            (*gradient.data_ptr)[i * gradient.strides[0] + gradient.offset] =
                p - (*target.data_ptr)[i * target.strides[0] + target.offset];
            loss -= (*target.data_ptr)[i * target.strides[0] + target.offset] * std::log(p);
        }
        return loss;
    }
};