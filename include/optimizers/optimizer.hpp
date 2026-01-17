#pragma once
#include <vector>
#include "../deep_learning/module.hpp"
#include "../deep_learning/network.hpp"
#include "../loss/Loss.hpp"

class Optimizer {
protected:
    Loss* loss;
    Network* model;
public:
    Optimizer(Network* model, Loss* loss) : model(model), loss(loss) {}
    virtual ~Optimizer() = default;
};