#pragma once
#include "optimizer.hpp"

class GD : public Optimizer {
protected:
    double learning_rate;
public:
    GD(Network* model, Loss* loss, double lr=0.01) : Optimizer(model, loss), learning_rate(lr) {}
    void step(const Tensor& LossDeriv) override;
    void train(const std::vector<Tensor>& X, const std::vector<Tensor>& y,
        const unsigned int epochs=10, unsigned int batch_size=100, const float lr=0.01f);
};