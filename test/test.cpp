// testing file

#include <iostream>
#include "../include/deep_learning/network.hpp"
#include "../include/deep_learning/linear.hpp"
#include "../include/optimizers/GD.hpp"
#include "../include/loss/MSE_Loss.hpp"

int main(){
    Network net(3, 4);
    std::shared_ptr<Linear> layer1 = std::make_shared<Linear>(3, 4);
    net.stackLayer(layer1);

    std::cout << "Network created with 1 layer.\n";

    std::vector< Tensor* > params = net.get_parameters();
    std::cout << "Initial Parameters:\n";
    for (auto param : params) {
        param->show();
    }
    for (size_t i = 0; i < params.size(); ++i){
        std::cout << "Parameter " << i << " shape: ";
        for (auto dim : params[i]->shape)
            std::cout << dim << " ";
        std::cout << "\n";
    }


    Tensor input({3}, { 1.0, -2.0, 3.0 });
    std::cout << "Input:\n";
    input.show();
    Tensor output = net.forward(input);
    std::cout << "Output:\n";
    output.show();

    MSELoss loss {4};
    Tensor target({4}, 2.0);

    float lossValue = loss.compute(output, target);
    std::cout << "Loss: " << lossValue << "\n";

    Tensor lossGrad = loss.get_gradient();
    GradientDescent optimizer(&net, 0.01);
    optimizer.step(lossGrad);

    params = net.get_parameters();
    std::cout << "Updated Parameters:\n";
    for (auto param : params) {
        param->show();
    }

    std::cout << "Step completed.\n";

    return 0;
}