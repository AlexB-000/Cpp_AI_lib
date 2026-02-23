#pragma once
#include <thread>
#include "../Tensor/cpp_tensor.hpp"
#include "Dataset.hpp"
#include "./deep_learning/network.hpp"

short bin_accuracy_sample(Network net, const Tensor *input, const Tensor *target){
    Tensor output = net.forward(*input);
    float predicted = output[0] >= 0.0f ? 1.0f : -1.0f;
    return predicted == (*target)[0] ? 1 : 0;
}

void bin_accuracy_thread(Network net, const std::vector<Tensor>* inputs, const std::vector<Tensor>* targets,
        std::vector<short>* results, unsigned int start_index, unsigned int count){
    for (unsigned int i=0; i<count; i++){
        (*results)[start_index + i] = bin_accuracy_sample(net, &(*inputs)[start_index + i], &(*targets)[start_index + i]);
     }
}

float bin_accuracy(Network& net, const std::vector<Tensor>& X, const std::vector<Tensor>& y){
    uint correct = 0;
    std::vector<short> thread_results(X.size());
    std::vector<std::thread> threads;

    uint32_t thread_count = std::thread::hardware_concurrency() - 1;
    unsigned int samples_per_thread = X.size() / thread_count + 1;

    for (uint32_t i=0; i<X.size(); i+= samples_per_thread){
        unsigned int batch_size = std::min(samples_per_thread, static_cast<uint32_t>(X.size() - i));
        threads.push_back(std::thread(bin_accuracy_thread, net, &X, &y, &thread_results, i, batch_size));
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

short multiclass_accuracy_sample(Network net, const Tensor *input, const Tensor *target){
    Tensor output = net.forward(*input);
    unsigned int predicted = 0;
    for (unsigned int j=1; j < output.shape[0]; j++){
        if (output[j] > output[predicted]){
            predicted = j;
        }
    }
    for (unsigned int j=0; j < target->shape[0]; j++){
        if ((*target)[j] == 1.0f && j == predicted){
            return 1;
        }
    }
    return 0;
}

void multiclass_accuracy_thread(Network net, const std::vector<Tensor>* inputs, const std::vector<Tensor>* targets,
        std::vector<short>* results, unsigned int start_index, unsigned int count){
    for (unsigned int i=0; i<count; i++){
        (*results)[start_index + i] = multiclass_accuracy_sample(net, &(*inputs)[start_index + i], &(*targets)[start_index + i]);
    }
}

float multiclass_accuracy(Network& net, const std::vector<Tensor>& X, const std::vector<Tensor>& y){
    uint correct = 0;
    std::vector<short> thread_results(X.size());
    std::vector<std::thread> threads;

    uint32_t thread_count = std::thread::hardware_concurrency() - 1;
    unsigned int samples_per_thread = X.size() / thread_count + 1;

    for (uint32_t i=0; i<X.size(); i+= samples_per_thread){
        unsigned int batch_size = std::min(samples_per_thread, static_cast<uint32_t>(X.size() - i));
        threads.push_back(std::thread(multiclass_accuracy_thread, net, &X, &y, &thread_results, i, batch_size));
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