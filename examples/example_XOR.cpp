#include <iostream>
#include <random>
#include "CppAI/preprocessing/train_test_split.hpp"
#include "CppAI/deep_learning/network.hpp"
#include "CppAI/deep_learning/linear.hpp"
#include "CppAI/optimizers/GD.hpp"
#include "CppAI/loss/BCEWithLogitsLoss.hpp"
#include "CppAI/deep_learning/ReLU.hpp"
#include "CppAI/activation/sigmoid.hpp"
#include "CppAI/evaluation.hpp"

std::vector<std::vector<Array<float>>> generate_data(unsigned int size){
    std::vector<Array<float>> data;
    std::vector<Array<float>> target;

    //random numbers generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (unsigned int i=0; i<size; i++){
        float value1 = dis(gen);
        float value2 = dis(gen);
        data.push_back(Array<float>{{2}, {value1, value2}});
        if (value1 && value2 || (!value1 && !value2))
            target.push_back(Array<float>{{1}, {0.0f}});
        else
            target.push_back(Array<float>{{1}, {1.0f}});
    }
    return {data, target};
}

void printDecisionGraph(Network& net){
    std::cout << std::fixed;
    std::cout.precision(2);
    std::cout << "## Decision Graph:\n";
    for (float x1 = 0.0f; x1 <= 1.0f; x1 += 0.2f){
        for (float x2 = 0.0f; x2 <= 1.0f; x2 += 0.2f){
            Array<float> input{{2}, {x1, x2}};
            Array<float> output = net.forward(input);
            std::cout << output[0] << " ";
        }
        std::cout << "\n";
    }
}

int main(){
    Network net;
    std::shared_ptr<Linear> layer1 = std::make_shared<Linear>(2, 5, 0.0f, 0.5f, 0.0f, 0.5f);
    net.stackLayer(layer1);
    std::shared_ptr<Sigmoid> activation1 = std::make_shared<Sigmoid>(5);
    net.stackLayer(activation1);
    std::shared_ptr<Linear> layer2 = std::make_shared<Linear>(5, 1, 0.0f, 0.5f, 0.0f, 0.5f);
    net.stackLayer(layer2);

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

    std::vector<std::vector<Array<float>>> splited = train_test_split(X, y, 0.1f);
    std::vector<Array<float>> X_train = splited[0];
    std::vector<Array<float>> y_train = splited[1];
    std::vector<Array<float>> X_test = splited[2];
    std::vector<Array<float>> y_test = splited[3];

    std::cout << "## Data generated and splitted. Training size: " << X_train.size() << ", Test size: " << X_test.size() << "\n";

    BCEWithLogitsLoss loss;
    GD optimizer(&net, &loss);

    std::cout << "## Starting training...\n";

    optimizer.train(X_train, y_train, 50, 100, 1.0f, 2, false);

    std::cout << "## Training completed.\n";


    std::shared_ptr<Sigmoid> activation2 = std::make_shared<Sigmoid>(1);
    net.stackLayer(activation2);

    params = net.get_parameters();
    std::cout << "Learned Parameters:\n";
    for (auto param : params) {
        param->show();
    }

    float train_accuracy = ClassMetrics::bin_accuracy(net, X_train, y_train);
    std::cout << "## Train Accuracy: " << train_accuracy * 100.0f << "%\n";

    float test_accuracy = ClassMetrics::bin_accuracy(net, X_test, y_test);
    std::cout << "## Test Accuracy: " << test_accuracy * 100.0f << "%\n";

    printDecisionGraph(net);

    return 0;
}