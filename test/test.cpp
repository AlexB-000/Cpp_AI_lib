// testing file

#include <iostream>
#include "../include/deep_learn/network.hpp"
#include "../include/deep_learn/linear.hpp"

int main(){
    Network net(3, 2);
    Linear layer1(3, 4);
    Linear layer2(4, 2);
    net.stackLayer(layer1);
    net.stackLayer(layer2);

    Tensor<double> input({3}, 1.0);
    Tensor<double> output = net.forward(input);
    std::cout << "Output:\n";
    output.show();

    Tensor<double> lossDeriv({2}, 1.0);
    std::vector<Tensor<double>> grads = net.backward(lossDeriv);
    std::cout << "Gradients:\n";
    for(auto& grad : grads){
        grad.show();
    }

    return 0;
}