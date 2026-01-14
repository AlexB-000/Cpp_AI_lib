#pragma once
#include "optimizer.hpp"

class GradientDescent : public Optimizer {
protected:
    double learning_rate;
public:
    GradientDescent(Network* model, double lr) : Optimizer(model), learning_rate(lr) {}
    void step(const Tensor& LossDeriv) override;
};