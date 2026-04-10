#pragma once
#include <iomanip>
#include <thread>
#include <mutex>
#include "optimizer.hpp"
#include "../DataLoader.hpp"

class GD : public Optimizer {
protected:
    std::mutex train_thread_mutex;

    inline void thread_step(Array<float>* param, const Array<float>* grad, const float lr);

    void step(const std::vector<Array<float>>& gradients, const float lr);

    std::vector<Array<float>> backpropagation(Network* _model, const Array<float>& LossDeriv);

public:
    GD(Network* model, Loss* loss) : Optimizer(model, loss) {}

    void train_oneSample(Network* _model, const Array<float>* input, const Array<float>* target, std::vector<Array<float>>* gradient, float* loss_value);

    void train_oneThread(Network _model, const std::vector<Array<float>>* inputs, const std::vector<Array<float>>* targets,
        std::vector<Array<float>>* gradient, float* loss_value, float grad_coeff,
        const unsigned int thread_idx, const unsigned int data_start_index, const unsigned int sample_count){
        
        std::vector<Array<float>> grad;
        float _inthread_loss_value = 0.0f;
        for (unsigned int i = 0; i < sample_count; ++i) {
            train_oneSample(&_model, &(*inputs)[data_start_index + i],
                            &(*targets)[data_start_index + i],
                            &grad, &_inthread_loss_value);
        }
        train_thread_mutex.lock();
        if (gradient->size() != 0) {
            for (size_t p = 0; p < grad.size(); ++p) {
                (*gradient)[p] += grad[p] * grad_coeff;
            }
        } else {
            *gradient = grad;
            for (size_t p = 0; p < grad.size(); ++p) {
                (*gradient)[p] *= grad_coeff;
            }
        }
        *loss_value += _inthread_loss_value;
        train_thread_mutex.unlock();
    }

    void train_oneThread_batch(Network _model, const std::vector<std::vector<Array<float>>>* batch,
        std::vector<Array<float>>* gradient, float* loss_value, float grad_coeff,
        const unsigned int thread_idx, const unsigned int start_index, const unsigned int sample_count){
        
        std::vector<Array<float>> grad;
        float _inthread_loss_value = 0.0f;
        for (unsigned int i = 0; i < sample_count; ++i) {
            train_oneSample(&_model, &(*batch)[start_index + i][0], &(*batch)[start_index + i][1], &grad, &_inthread_loss_value);
        }
        train_thread_mutex.lock();
        if (gradient->size() != 0) {
            for (size_t p = 0; p < grad.size(); ++p) {
                (*gradient)[p] += grad[p] * grad_coeff;
            }
        } else {
            *gradient = grad;
            for (size_t p = 0; p < grad.size(); ++p) {
                (*gradient)[p] *= grad_coeff;
            }
        }
        *loss_value += _inthread_loss_value;
        train_thread_mutex.unlock();
    }

    void train(const std::vector<Array<float>>& X, const std::vector<Array<float>>& y,
        const unsigned int epochs=10, unsigned int batch_size=100, const float lr=0.01f,
        short display_mode=0, bool multithreading=true, bool flat_out=false);

    void train(DataLoader& dataloader, const unsigned int epochs=10, const float lr=0.01f,
        short display_mode=0, bool multithreading=true, bool flat_out=false);
};