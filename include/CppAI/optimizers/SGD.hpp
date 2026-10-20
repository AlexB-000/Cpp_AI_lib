#pragma once
#include <iomanip>
#include <thread>
#include <mutex>
#include "optimizer.hpp"
#include "../DataLoader.hpp"

class SGD : public Optimizer {
protected:
    std::mutex train_thread_mutex;

    float W_decay;
    float _momentum;

    inline void thread_step(Array<float>* param, const Array<float>* grad, const float lr);

    void step(std::vector<Array<float>>& velocity, std::vector<Array<float>>& gradient, const float lr);
public:
    SGD(Network* model, Loss* loss, float weight_decay = 0.0f, float momentum = 0.0f)
        : Optimizer(model, loss),
          W_decay(weight_decay),
          _momentum(momentum)
    {
    }

    void train_oneSample(Network* _model, const Array<float>* input, const Array<float>* target, std::vector<Array<float>>* gradient, float* loss_value);

    void train_oneThread(Network _model, const std::vector<Array<float>>* inputs, const std::vector<Array<float>>* targets,
        std::vector<Array<float>>* gradient, float* loss_value, float grad_coeff,
        const unsigned int thread_idx, const unsigned int data_start_index, const unsigned int sample_count);

    void train_oneThread_batch(Network _model, const std::vector<std::vector<Array<float>>>* batch,
        std::vector<Array<float>>* gradient, float* loss_value, float grad_coeff,
        const unsigned int thread_idx, const unsigned int start_index, const unsigned int sample_count);

    void train(const std::vector<Array<float>>& X, const std::vector<Array<float>>& y,
        const unsigned int epochs=10, unsigned int batch_size=100, const float lr=0.01f,
        short display_mode=0, bool multithreading=true, bool flat_out=false);

    void train(DataLoader& dataloader, const unsigned int epochs=10, const float lr=0.01f,
        short display_mode=0, bool multithreading=true, bool flat_out=false);
};