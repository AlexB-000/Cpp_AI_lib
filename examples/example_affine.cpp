#include <iostream>
#include <random>
#include "../include/preprocessing/train_test_split.hpp"
#include "../include/deep_learning/network.hpp"
#include "../include/deep_learning/linear.hpp"
#include "../include/optimizers/GD.hpp"
#include "../include/loss/MSE_Loss.hpp"

std::vector<std::vector<Array<float>>> generate_data(uint size){
    std::vector<Array<float>> data;
    std::vector<Array<float>> target;

    //random numbers generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 10);

    for (uint i=0; i<size; i++){
        float value = dis(gen)/10.0f;
        data.push_back(Array<float>{{1}, {value}});
        target.push_back(Array<float>{{1}, {value*2 + 1}});
    }
    return {data, target};
}

int main(){
    Network net;
    std::shared_ptr<Linear> layer1 = std::make_shared<Linear>(1, 1);
    net.stackLayer(layer1);

    std::cout << "## Network created.\n";

    std::vector< Array<float>* > params = net.get_parameters();
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


    std::vector<std::vector<Array<float>>> data = generate_data(1000);
    std::vector<Array<float>> X = data[0];
    std::vector<Array<float>> y = data[1];

    std::vector<std::vector<Array<float>>> splited = train_test_split(X, y, 0.1);
    std::vector<Array<float>> X_train = splited[0];
    std::vector<Array<float>> y_train = splited[1];
    std::vector<Array<float>> X_test = splited[2];
    std::vector<Array<float>> y_test = splited[3];

    std::cout << "## Data generated and splitted. Training size: " << X_train.size() << ", Test size: " << X_test.size() << "\n";

    MSELoss loss {1};
    GD optimizer(&net, &loss);

    std::cout << "## Starting training...\n";

    optimizer.train(X_train, y_train, 10, 100, 0.7, true);

    std::cout << "## Training completed.\n";

    params = net.get_parameters();
    std::cout << "Learned Parameters:\n";
    for (auto param : params) {
        param->show();
    }

    return 0;
}