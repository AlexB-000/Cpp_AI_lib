#pragma once
#include "../Tensor/cpp_tensor.hpp"
#include "Dataset.hpp"
#include "./deep_learning/network.hpp"

float bin_accuracy(Network& net, const std::vector<Tensor>& X, const std::vector<Tensor>& y){
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

float bin_accuracy(Network& net, const Dataset& dataset){
    std::vector<Tensor> X, y;

    for (unsigned int i=0; i<dataset.size(); i++){
        std::vector<Tensor> sample = dataset[i];
        X.push_back(sample[0]);
        y.push_back(sample[1]);
    }
    return bin_accuracy(net, X, y);
}

float multiclass_accuracy(Network& net, const std::vector<Tensor>& X, const std::vector<Tensor>& y){
    uint correct = 0;
    for (unsigned int i=0; i<X.size(); i++){
        Tensor output = net.forward(X[i]);
        unsigned int predicted = 0;
        for (unsigned int j=1; j<output.shape[0]; j++){
            if (output[j] > output[predicted]){
                predicted = j;
            }
        }
        for (unsigned int j=0; j<y[i].shape[0]; j++){
            if (y[i][j] == 1.0f && j == predicted){
                correct++;
                break;
            }
        }
    }
    return static_cast<float>(correct) / static_cast<float>(X.size());
}

float multiclass_accuracy(Network& net, const Dataset& dataset){
    std::vector<Tensor> X, y;

    for (unsigned int i=0; i<dataset.size(); i++){
        std::vector<Tensor> sample = dataset[i];
        X.push_back(sample[0]);
        y.push_back(sample[1]);
    }
    return multiclass_accuracy(net, X, y);
}