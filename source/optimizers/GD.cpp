#include "../../include/optimizers/GD.hpp"

void GD::thread_step(Tensor* param, const Tensor* grad, const float lr) {
    (*param) = (*param) - (*grad) * lr;
}

void GD::step(const std::vector<Tensor>& gradients, const float lr) {
    
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

std::vector<Tensor> GD::backpropagation(const Tensor& LossDeriv) {
    Tensor deriv = LossDeriv;
    std::vector<Tensor> all_grads;

    for (int layer_idx = model->modules.size()-1; layer_idx >= 0; layer_idx--) {
        std::shared_ptr<Module> layer = model->modules[layer_idx];

        // std::cout << "Backpropagating through layer : " << layer << " index: " << layer_idx << "\n";

        std::vector<Tensor> grad = layer->backward(deriv);

        // std::cout << "Layer backward gradients:\n";
        // for (size_t i = 0; i < grad.size(); ++i) {
        //     grad[i].show();
        // }
        
        // Skip updating parameters that do not have gradients (non-trainable parameters)
        // their tensor is empty so it represents the scalar 0.0f by default
        if (grad[0].dim() > 0) {
            all_grads.insert(all_grads.begin(), grad.begin(), grad.end()-1);
        }
        deriv = grad.back();
    }
    return all_grads;
}

void GD::train_oneSample(const Tensor* input, const Tensor* target, std::vector<Tensor>* gradient, float* loss_value) {
    Tensor output = model->forward(*input);
    
    *loss_value += loss->compute(output, *target);

    Tensor lossGrad = loss->get_gradient();

    std::vector<Tensor> grad = backpropagation(lossGrad);

    if (gradient->size() != 0) {
        for (size_t p = 0; p < grad.size(); ++p) {
            (*gradient)[p] = (*gradient)[p] + grad[p];
        }
    } else {
        *gradient = grad;
    }
}

void GD::train(const std::vector<Tensor>& X, const std::vector<Tensor>& y,
    const unsigned int epochs, unsigned int batch_size, const float lr,
    bool show_progress, bool show_batch_progress, bool flat_out) {
    
    unsigned int num_samples = X.size();

    std::vector<Tensor> batch_gradient;

    uint32_t max_thread_count;
    if (flat_out) max_thread_count = std::thread::hardware_concurrency();
    else max_thread_count = std::thread::hardware_concurrency() - 1;

    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "--Epoch " << epoch + 1 << "/" << epochs << "\n";

        float epoch_loss = 0.0;

        for (unsigned int i = 0; i < num_samples; i += batch_size) {
            batch_gradient.clear();

            unsigned int actual_batch_size = std::min(batch_size, num_samples - i);

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
                const unsigned int data_start_index = i + t * samples_per_thread;
                const unsigned int count = std::min(samples_per_thread, actual_batch_size - t * samples_per_thread);
                threads.push_back(std::thread(&GD::train_oneThread, this,
                    &X, &y, &batch_gradient, &epoch_loss, 1.0f / static_cast<float>(actual_batch_size),
                    t, data_start_index, count));
            }
            for (std::thread& t : threads) {
                t.join();
            }
            
            step(batch_gradient, lr);

            if (show_batch_progress) {
                std::cout << "  batch completed " << (i / batch_size) + 1 << "/" << (num_samples / batch_size + 1) << "\n";
            }
        }

        epoch_loss /= static_cast<float>(num_samples);

        std::cout << "--Epoch completed " << epoch + 1 << "/" << epochs << ", Loss: " << epoch_loss << std::endl;

        if (show_progress) {
            std::vector<Tensor*> params = model->get_parameters();
            std::cout << "Epoch Parameters:\n";
            for (auto param : params) {
                param->show();
            }
        }
    }
}


void GD::train(DataLoader& data_loader, const unsigned int epochs, const float lr,
    bool show_progress, bool show_batch_progress, bool flat_out) {
    
    std::vector<Tensor> batch_gradient;

    uint32_t max_thread_count;
    if (flat_out) max_thread_count = std::thread::hardware_concurrency();
    else max_thread_count = std::thread::hardware_concurrency() - 1;

    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "--Epoch " << epoch + 1 << "/" << epochs << "\n";

        float epoch_loss = 0.0;

        for (unsigned int i = 0; i < data_loader.batch_quantity(); i++) {
            batch_gradient.clear();

            std::vector<std::vector<Tensor>> batch = data_loader.get_batch(i);
            
            unsigned int batch_size = batch.size();

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
                unsigned int start_index = t * samples_per_thread;
                unsigned int count = std::min(samples_per_thread, batch_size - start_index);
                
                threads.push_back(std::thread(&GD::train_oneThread_batch, this,
                    &batch, &batch_gradient, &epoch_loss, 1.0f / static_cast<float>(batch_size),
                    t, start_index, count));
            }
            for (auto& t : threads) {
                t.join();
            }

            step(batch_gradient, lr);

            if (show_batch_progress) {
                std::cout << "  batch completed " << i + 1 << "/" << data_loader.batch_quantity() << "\n";
            }
        }

        epoch_loss /= static_cast<float>(data_loader.dataset_size());

        std::cout << "--Epoch completed " << epoch + 1 << "/" << epochs << ", Loss: " << epoch_loss << std::endl;

        if (show_progress) {
            std::vector<Tensor*> params = model->get_parameters();
            std::cout << "Epoch Parameters:\n";
            for (auto param : params) {
                param->show();
            }
        }
    }
}