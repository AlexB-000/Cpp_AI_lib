#pragma once
#include <thread>
#include "optimizer.hpp"
#include "../DataLoader.hpp"

class GD : public Optimizer {
protected:
    void step(const std::vector<Tensor>& gradients, const float lr);

    std::vector<Tensor> backpropagation(const Tensor& LossDeriv);

public:
    GD(Network* model, Loss* loss) : Optimizer(model, loss) {}

    void train_oneSample(const Tensor* input, const Tensor* target, std::vector<Tensor>* gradient, float* loss_value);

    void train_oneThread(const std::vector<Tensor>* inputs, const std::vector<Tensor>* targets,
        std::vector<std::vector<Tensor>>* gradients, std::vector<float>* loss_values,
        const unsigned int data_start_index, const unsigned int result_start_index, const unsigned int sample_count){
        for (unsigned int i = 0; i < sample_count; ++i) {
            train_oneSample(&(*inputs)[data_start_index + i],
                            &(*targets)[data_start_index + i],
                            &(*gradients)[result_start_index + i], &(*loss_values)[result_start_index + i]);
        }
    }

    void train_oneThread_batch(const std::vector<std::vector<Tensor>>* batch,
        std::vector<std::vector<Tensor>>* gradients, std::vector<float>* loss_values,
        const unsigned int start_index, const unsigned int sample_count){
        for (unsigned int i = 0; i < sample_count; ++i) {
            train_oneSample(&(*batch)[start_index + i][0], &(*batch)[start_index + i][1], &(*gradients)[start_index + i], &(*loss_values)[start_index + i]);
        }
    }

    void train(const std::vector<Tensor>& X, const std::vector<Tensor>& y,
        const unsigned int epochs=10, unsigned int batch_size=100, const float lr=0.01f,
        bool show_progress=false, bool show_batch_progress=false);

    void train(DataLoader& dataloader, const unsigned int epochs=10, const float lr=0.01f,
        bool show_progress=false, bool show_batch_progress=false);
};