#pragma once
#include <vector>
#include "../deep_learning/module.hpp"
#include "../deep_learning/network.hpp"

class Optimizer {
protected:
    Network* model;
public:
    Optimizer(Network* model) : model(model) {}
    virtual ~Optimizer() = default;
    virtual void step(const Tensor& LossDeriv) = 0;
};