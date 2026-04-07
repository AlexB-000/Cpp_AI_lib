#include "CppAI/optimizers/GD.hpp"

void GD::thread_step(Array<float>* param, const Array<float>* grad, const float lr) {
    (*param) -= (*grad) * lr;
}

void GD::step(const std::vector<Array<float>>& gradients, const float lr) {
    
    // uint32_t thread_count = std::thread::hardware_concurrency() - 1;

    // if (thread_count > model->parameters.size()) {
    //     thread_count = model->parameters.size();
    // }

    uint32_t thread_count = model->parameters.size();

    std::vector<std::thread> threads;

    for (int i = 0; i < model->parameters.size(); ++i) {
        threads.push_back(std::thread(&GD::thread_step, this, model->parameters[i], &gradients[i], lr));
    }
    for (std::thread& t : threads) {
        t.join();
    }
}

std::vector<Array<float>> GD::backpropagation(Network* _model, const Array<float>& LossDeriv) {
    Array<float> deriv = LossDeriv;
    std::vector<Array<float>> all_grads;

    for (int layer_idx = _model->modules.size()-1; layer_idx >= 0; layer_idx--) {
        std::shared_ptr<Module> layer = _model->modules[layer_idx];

        // std::cout << "Backpropagating through layer : " << layer << " index: " << layer_idx << "\n";

        std::vector<Array<float>> grad = layer->backward(deriv);

        // std::cout << "Layer backward gradients:\n";
        // for (size_t i = 0; i < grad.size(); ++i) {
        //     grad[i].show();
        // }
        
        // Skip updating parameters that do not have gradients (non-trainable parameters)
        // their Array<float> is empty so it represents the scalar 0.0f by default
        if (grad[0].dim > 0) {
            all_grads.insert(all_grads.begin(), grad.begin(), grad.end()-1);
        }
        deriv = grad.back();
    }
    return all_grads;
}

void GD::train_oneSample(Network* _model, const Array<float>* input, const Array<float>* target, std::vector<Array<float>>* gradient, float* loss_value) {
    Array<float> output = _model->forward(*input);
    
    *loss_value += loss->compute(output, *target);

    Array<float> lossGrad = loss->get_gradient();

    std::vector<Array<float>> grad = backpropagation(_model, lossGrad);

    if (gradient->size() != 0) {
        for (size_t p = 0; p < grad.size(); ++p) {
            (*gradient)[p] += grad[p];
        }
    } else {
        *gradient = grad;
    }
}

void GD::train(const std::vector<Array<float>>& X, const std::vector<Array<float>>& y,
    const unsigned int epochs, unsigned int batch_size, const float lr,
    short display_mode, bool multithreading, bool flat_out) {
    
    model->train_mode();
    unsigned int num_samples = X.size();

    std::vector<Array<float>> batch_gradient;

    uint32_t max_thread_count;
    if (flat_out) max_thread_count = std::thread::hardware_concurrency();
    else max_thread_count = std::thread::hardware_concurrency() - 1;
    if (!multithreading) max_thread_count = 1;

    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        if (display_mode) {
            std::cout << "--Epoch " << epoch + 1 << "/" << epochs << "\n";
        }
        float epoch_loss = 0.0;

        for (unsigned int i = 0; i < num_samples; i += batch_size) {
            batch_gradient.clear();

            unsigned int actual_batch_size = std::min(batch_size, num_samples - i);
            const float grad_coeff = 1.0f / static_cast<float>(actual_batch_size);

            uint32_t thread_count;
            if (actual_batch_size < max_thread_count) {
                // If the batch size is smaller than the number of threads, reduce the thread count
                thread_count = actual_batch_size;
            } else {
                thread_count = max_thread_count;
            }

            std::vector<std::thread> threads;
            unsigned int samples_per_thread = (batch_size + thread_count - 1) / thread_count;
            
            for (unsigned int t = 0; t < thread_count; ++t) {
                if (t * samples_per_thread >= actual_batch_size) {
                    break; // No more samples to process
                }
                const unsigned int data_start_index = i + t * samples_per_thread;
                const unsigned int count = std::min(samples_per_thread, actual_batch_size - t * samples_per_thread);
                threads.push_back(std::thread(&GD::train_oneThread, this, *model,
                    &X, &y, &batch_gradient, &epoch_loss, grad_coeff,
                    t, data_start_index, count));
            }
            for (std::thread& t : threads) {
                t.join();
            }
            
            step(batch_gradient, lr);

            if (display_mode == 3) {
                std::cout << "  batch completed " << (i / batch_size) + 1 << "/" << (num_samples / batch_size + 1) << "\n";
            }
        }

        epoch_loss /= static_cast<float>(num_samples);

        if (display_mode) {
            std::cout << "--Epoch completed " << epoch + 1 << "/" << epochs << ", Loss: " << epoch_loss << std::endl;

            if (display_mode == 2) {
                std::vector<Array<float>*> params = model->get_parameters();
                std::cout << "Epoch Parameters:\n";
                for (auto param : params) {
                    param->show();
                }
            }
        }
    }
    model->eval_mode();
}


void GD::train(DataLoader& data_loader, const unsigned int epochs, const float lr,
    short display_mode, bool multithreading, bool flat_out) {
    
    model->train_mode();
    std::vector<Array<float>> batch_gradient;

    uint32_t max_thread_count;
    if (flat_out) max_thread_count = std::thread::hardware_concurrency();
    else max_thread_count = std::thread::hardware_concurrency() - 1;
    if (!multithreading) max_thread_count = 1;

    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        if (display_mode) {
            std::cout << "--Epoch " << epoch + 1 << "/" << epochs << "\n";
        }

        float epoch_loss = 0.0;

        for (unsigned int i = 0; i < data_loader.batch_quantity(); i++) {
            batch_gradient.clear();

            std::vector<std::vector<Array<float>>> batch = data_loader.get_batch(i);
            
            unsigned int batch_size = batch.size();

            const float grad_coeff = 1.0f / static_cast<float>(batch_size);

            uint32_t thread_count;
            if (batch_size < max_thread_count) {
                // If the batch size is smaller than the number of threads, reduce the thread count
                thread_count = batch_size;
            } else {
                thread_count = max_thread_count;
            }

            std::vector<std::thread> threads;
            unsigned int samples_per_thread = (batch_size + thread_count - 1) / thread_count;
            
            for (unsigned int t = 0; t < thread_count; ++t) {
                if (t * samples_per_thread >= batch_size) {
                    break; // No more samples to process
                }
                unsigned int start_index = t * samples_per_thread;
                unsigned int count = std::min(samples_per_thread, batch_size - start_index);
                
                threads.push_back(std::thread(&GD::train_oneThread_batch, this, *model,
                    &batch, &batch_gradient, &epoch_loss, grad_coeff,
                    t, start_index, count));
            }
            for (auto& t : threads) {
                t.join();
            }

            step(batch_gradient, lr);

            if (display_mode == 3) {
                std::cout << "  batch completed " << i + 1 << "/" << data_loader.batch_quantity() << "\n";
            }
        }

        epoch_loss /= static_cast<float>(data_loader.dataset_size());

        if (display_mode) {
            std::cout << "--Epoch completed " << epoch + 1 << "/" << epochs << ", Loss: " << epoch_loss << std::endl;

            if (display_mode == 2) {
                std::vector<Array<float>*> params = model->get_parameters();
                std::cout << "Epoch Parameters:\n";
                for (auto param : params) {
                    param->show();
                }
            }
        }
    }
    model->eval_mode();
}