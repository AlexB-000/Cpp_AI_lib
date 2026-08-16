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
        float loss = 0;
        for (uint32_t i=0; i<gradient.shape[0]; i++){
            (*gradient.data_ptr)[i*gradient.strides[0] + gradient.offset] = -(*target.data_ptr)[i*target.strides[0] + target.offset]
                                      / ((*predicted.data_ptr)[i*predicted.strides[0] + predicted.offset] + _epsilon);
            loss -= (*target.data_ptr)[i*target.strides[0] + target.offset]
                    * std::log((*predicted.data_ptr)[i*predicted.strides[0] + predicted.offset] + _epsilon);
        }
        return loss;
    }
};