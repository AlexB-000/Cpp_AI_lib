#pragma once
#include "optimizer.hpp"
#include "../DataLoader.hpp"

class GD : public Optimizer {
protected:
    void step(const std::vector<Tensor>& gradients, const float lr);

    std::vector<Tensor> backpropagation(const Tensor& LossDeriv);

public:
    GD(Network* model, Loss* loss) : Optimizer(model, loss) {}

    void train(const std::vector<Tensor>& X, const std::vector<Tensor>& y,
        const unsigned int epochs=10, unsigned int batch_size=100, const float lr=0.01f, bool show_progress=false, bool show_batch_progress=false);

    void train(DataLoader& dataloader, const unsigned int epochs=10, const float lr=0.01f, bool show_progress=false, bool show_batch_progress=false);
};