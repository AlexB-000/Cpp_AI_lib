#include <iostream>
#include <random>
#include "../include/preprocessing/train_test_split.hpp"
#include "../include/deep_learning/network.hpp"
#include "../include/deep_learning/linear.hpp"
#include "../include/optimizers/GD.hpp"
#include "../include/loss/MSE_Loss.hpp"
#include "../include/deep_learning/ReLU.hpp"
#include "../include/deep_learning/tanh.hpp"

std::vector<std::vector<Tensor>> generate_data(uint size){
    std::vector<Tensor> data;
    std::vector<Tensor> target;

    //random numbers generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (uint i=0; i<size; i++){
        float value1 = dis(gen);
        float value2 = dis(gen);
        data.push_back(Tensor{{2}, {value1, value2}});
        if (value1 && value2 || (!value1 && !value2))
            target.push_back(Tensor{{1}, {0.0f}});
        else
            target.push_back(Tensor{{1}, {1.0f}});
    }
    return {data, target};
}

int main(){
    Network net(2, 1);
    std::shared_ptr<Linear> layer1 = std::make_shared<Linear>(2, 2, 0.5f, 1.0f);
    net.stackLayer(layer1);
    std::shared_ptr<ReLU> activation1 = std::make_shared<ReLU>(2, 2);
    net.stackLayer(activation1);
    std::shared_ptr<Linear> layer2 = std::make_shared<Linear>(2, 1, 0.5f, 1.0f);
    net.stackLayer(layer2);
    std::shared_ptr<Tanh> activation2 = std::make_shared<Tanh>(1, 1);
    net.stackLayer(activation2);

    std::cout << "## Network created.\n";

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


    std::vector<std::vector<Tensor>> data = generate_data(1000);
    std::vector<Tensor> X = data[0];
    std::vector<Tensor> y = data[1];

    std::vector<std::vector<Tensor>> splited = train_test_split(X, y, 0.1);
    std::vector<Tensor> X_train = splited[0];
    std::vector<Tensor> y_train = splited[1];
    std::vector<Tensor> X_test = splited[2];
    std::vector<Tensor> y_test = splited[3];

    std::cout << "## Data generated and splitted. Training size: " << X_train.size() << ", Test size: " << X_test.size() << "\n";

    MSELoss loss {1};
    GD optimizer(&net, &loss);

    std::cout << "## Starting training...\n";

    optimizer.train(X_train, y_train, 10, 100, 0.65);

    std::cout << "## Training completed.\n";

    params = net.get_parameters();
    std::cout << "Learned Parameters:\n";
    for (auto param : params) {
        param->show();
    }

    return 0;
}