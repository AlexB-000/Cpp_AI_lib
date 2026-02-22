#pragma once
#include "Dataset.hpp"
#include "./deep_learning/network.hpp"

float accuracy(Network& net, const std::vector<Tensor>& X, const std::vector<Tensor>& y){
    uint correct = 0;
    for (unsigned int i=0; i<X.size(); i++){
        Tensor output = net.forward(X[i]);
        float predicted = output[0] >= 0.0f ? 1.0f : -1.0f;
        if (predicted == y[i][0]){
            correct++;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(X.size());
}

float accuracy(Network& net, const Dataset& dataset){
    uint correct = 0;
    for (unsigned int i=0; i<dataset.size(); i++){
        std::vector<Tensor>data = dataset[i];

        Tensor output = net.forward(data[0]);
        float predicted = output[0] >= 0.0f ? 1.0f : -1.0f;
        if (predicted == data[1][0]){
            correct++;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(dataset.size());
}