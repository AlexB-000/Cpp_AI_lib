#include "../../include/optimizers/GD.hpp"

void GD::step(const Tensor& LossDeriv) {
    // std::cout << "Parameters before update:\n";
    // std::vector<Tensor*> params_before = model->get_parameters();
    // for (auto param : params_before) {
    //     param->show();
    // }

    Tensor deriv = LossDeriv;
    for (std::shared_ptr<Module> layer : model->modules) {
        auto grad = layer->backward(deriv);
        auto layerParams = layer->get_parameters();

        // std::cout << "Gradients:\n";
        // for (size_t i = 0; i < grad.size(); ++i) {
        //     grad[i].show();
        // }

        for (size_t i = 0; i < layerParams.size(); ++i) {
            *(layerParams[i]) = *(layerParams[i]) - grad[i] * learning_rate;
        }
        deriv = grad.back();
    }
}

void GD::train(const std::vector<Tensor>& X, const std::vector<Tensor>& y,
    const unsigned int epochs, unsigned int batch_size, const float lr) {
    
    learning_rate = lr;
    unsigned int num_samples = X.size();

    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "--Epoch " << epoch + 1 << "/" << epochs << "\n";

        double epoch_loss = 0.0;

        for (unsigned int i = 0; i < num_samples; i += batch_size) {
            unsigned int actual_batch_size = std::min(batch_size, num_samples - i);

            for (unsigned int j = i; j < i+actual_batch_size; ++j) {
                std::cout << "  --Sample " << j + 1 << "/" << num_samples << "\n";

                Tensor output = model->forward(X[j]);

                std::cout << "   --Input: ";
                X[j].show();
                std::cout << "   --Output: ";
                output.show();
                std::cout << "   --Target: ";
                y[j].show();
                float loss_value = loss->compute(output, y[j]);
                
                epoch_loss += loss_value;

                Tensor lossGrad = loss->get_gradient();

                // std::cout << "   --Loss: " << loss_value << "\n";
                // std::cout << "   --Loss Gradient: ";
                // lossGrad.show();

                step(lossGrad);
            }

        }
        epoch_loss /= static_cast<float>(num_samples);

        std::cout << "--Epoch completed " << epoch + 1 << "/" << epochs << ", Loss: " << epoch_loss << std::endl;

        std::vector<Tensor*> params = model->get_parameters();
        std::cout << "Epoch Parameters:\n";
        for (auto param : params) {
            param->show();
        }
    }
}