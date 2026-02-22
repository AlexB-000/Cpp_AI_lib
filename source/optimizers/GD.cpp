#include "../../include/optimizers/GD.hpp"

void GD::step(const std::vector<Tensor>& gradients, const float lr) {
    // std::cout << "Parameters before update:\n";
    // std::vector<Tensor*> params_before = model->get_parameters();
    // for (auto param : params_before) {
    //     param->show();
    // }

    // std::cout << "Gradients:\n";
    // for (size_t i = 0; i < gradients.size(); ++i) {
    //     gradients[i].show();
    // }

    for (int i = 0; i < model->parameters.size(); ++i) {
        *(model->parameters[i]) = *(model->parameters[i]) - gradients[i] * lr;
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

void GD::train(const std::vector<Tensor>& X, const std::vector<Tensor>& y,
    const unsigned int epochs, unsigned int batch_size, const float lr,
    bool show_progress, bool show_batch_progress) {
    
    unsigned int num_samples = X.size();

    std::vector<Tensor> batch_gradient;

    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "--Epoch " << epoch + 1 << "/" << epochs << "\n";

        double epoch_loss = 0.0;

        for (unsigned int i = 0; i < num_samples; i += batch_size) {
            batch_gradient.clear();

            unsigned int actual_batch_size = std::min(batch_size, num_samples - i);

            for (unsigned int j = i; j < i+actual_batch_size; ++j) {
                // std::cout << "  --Sample " << j + 1 << "/" << num_samples << "\n";

                Tensor output = model->forward(X[j]);

                // std::cout << "   --Input: ";
                // X[j].show();
                // std::cout << "   --Output: ";
                // output.show();
                // std::cout << "   --Target: ";
                // y[j].show();

                float loss_value = loss->compute(output, y[j]);
                
                epoch_loss += loss_value;

                Tensor lossGrad = loss->get_gradient();

                // std::cout << "   --Loss: " << loss_value << "\n";
                // std::cout << "   --Loss Gradient: ";
                // lossGrad.show();

                if (j == i) {
                    batch_gradient = backpropagation(lossGrad);
                } else {
                    std::vector<Tensor> sample_grads = backpropagation(lossGrad);
                    for (size_t k = 0; k < batch_gradient.size(); ++k) {
                        batch_gradient[k] = batch_gradient[k] + sample_grads[k];
                    }
                }
            }

            for (size_t k = 0; k < batch_gradient.size(); ++k) {
                batch_gradient[k] = batch_gradient[k] * (1.0f / static_cast<float>(actual_batch_size));
            }
            // std::cout << "  --Batch gradient:\n";
            // for (size_t k = 0; k < batch_gradient.size(); ++k) {
            //     batch_gradient[k].show();
            // }
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
    bool show_progress, bool show_batch_progress) {
    
    std::vector<Tensor> batch_gradient;

    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "--Epoch " << epoch + 1 << "/" << epochs << "\n";

        double epoch_loss = 0.0;

        for (unsigned int i = 0; i < data_loader.batch_quantity(); i++) {
            batch_gradient.clear();

            std::vector<std::vector<Tensor>> batch = data_loader.get_batch(i);
            
            unsigned int batch_size = batch.size();

            for (unsigned int j = 0; j < batch_size; ++j) {
                Tensor output = model->forward(batch[j][0]);
                
                float loss_value = loss->compute(output, batch[j][1]);
                
                epoch_loss += loss_value;

                Tensor lossGrad = loss->get_gradient();
                
                if (j == 0) {
                    batch_gradient = backpropagation(lossGrad);
                } else {
                    std::vector<Tensor> sample_grads = backpropagation(lossGrad);
                    for (size_t k = 0; k < batch_gradient.size(); ++k) {
                        batch_gradient[k] = batch_gradient[k] + sample_grads[k];
                    }
                }
            }

            for (size_t k = 0; k < batch_gradient.size(); ++k) {
                batch_gradient[k] = batch_gradient[k] * (1.0f / static_cast<float>(batch_size));
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