#pragma once
#include <thread>
#include "../Tensor/cpp_tensor.hpp"
#include "Dataset.hpp"
#include "./deep_learning/network.hpp"

void bin_accuracy_thread(Network net, const Tensor* input, const Tensor* target, short* result){
    Tensor output = net.forward(*input);
    float predicted = output[0] >= 0.0f ? 1.0f : -1.0f;
    *result = predicted == (*target)[0] ? 1 : 0;
}

float bin_accuracy(Network& net, const std::vector<Tensor>& X, const std::vector<Tensor>& y){
    uint correct = 0;
    std::vector<short> thread_results(X.size());
    std::vector<std::thread> threads;
    for (unsigned int i=0; i<X.size(); i++){
        threads.push_back(std::thread(bin_accuracy_thread, net, &X[i], &y[i], &thread_results[i]));
    }
    for (std::thread& t : threads){
        t.join();
    }
    for (short result : thread_results){
        correct += result;
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

void multiclass_accuracy_thread(Network net, const Tensor* input, const Tensor* target, short* result){
    Tensor output = net.forward(*input);
    unsigned int predicted = 0;
    for (unsigned int j=1; j < output.shape[0]; j++){
        if (output[j] > output[predicted]){
            predicted = j;
        }
    }
    for (unsigned int j=0; j < (*target).shape[0]; j++){
        if ((*target)[j] == 1.0f && j == predicted){
            *result = 1;
            return;
        }
    }
    *result = 0;
}

float multiclass_accuracy(Network& net, const std::vector<Tensor>& X, const std::vector<Tensor>& y){
    uint correct = 0;
    std::vector<short> thread_results(X.size());
    std::vector<std::thread> threads;
    for (unsigned int i=0; i<X.size(); i++){
        threads.push_back(std::thread(multiclass_accuracy_thread, net, &X[i], &y[i], &thread_results[i]));
    }
    for (std::thread& t : threads){
        t.join();
    }
    for (short result : thread_results){
        correct += result;
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