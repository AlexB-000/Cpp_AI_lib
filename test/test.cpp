// testing file

#include <iostream>
#include "../include/deep_learn/network.hpp"
#include "../include/deep_learn/linear.hpp"

int main(){
    Network net(3, 2);
    std::shared_ptr<Linear> layer1 = std::make_shared<Linear>(3, 4);
    std::shared_ptr<Linear> layer2 = std::make_shared<Linear>(4, 2);
    net.stackLayer(layer1);
    net.stackLayer(layer2);

    Tensor input({3}, 1.0);
    Tensor output = net.forward(input);
    std::cout << "Output:\n";
    output.show();

    Tensor lossDeriv({2}, 1.0);
    std::vector<Tensor> grads = net.backward(lossDeriv);
    std::cout << "Gradients:\n";
    for(auto& grad : grads){
        grad.show();
    }

    return 0;
}