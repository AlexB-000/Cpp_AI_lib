#include "../../include/optimizers/GD.hpp"

void GradientDescent::step(const Tensor& LossDeriv) {
    Tensor deriv = LossDeriv;
    for (std::shared_ptr<Module> layer : model->modules) {
        auto grad = layer->backward(deriv);
        auto layerParams = layer->get_parameters();

        for (size_t i = 0; i < layerParams.size(); ++i) {
            *(layerParams[i]) = *(layerParams[i]) - grad[i] * learning_rate;
        }
        deriv = grad.back();
    }
}