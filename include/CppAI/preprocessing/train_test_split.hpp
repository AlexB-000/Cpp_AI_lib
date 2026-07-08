#pragma once
#include <random>
#include <vector>
#include "Arrays/ndarrays.hpp"

std::vector<std::vector<Array<float>>> train_test_split(std::vector<Array<float>> X, std::vector<Array<float>> y,
    float test_prop=0.1, int32_t seed=-1) {
    
    std::vector<Array<float>> X_train, y_train, X_test, y_test;
    
    //random numbers generator
    if (seed == -1) {
        std::random_device rd;
        seed = rd();
    }
    std::mt19937 gen(seed);
    
    unsigned int test_size = X.size() * test_prop;
    for (unsigned int i = 0; i<test_size; i++){
        unsigned int data_size = X.size();
        std::uniform_int_distribution<> dis(0, data_size - 1);
        unsigned int idx = dis(gen);
        X_test.emplace_back(X[idx]);
        y_test.emplace_back(y[idx]);

        std::swap(X[idx], X.back());
        std::swap(y[idx], y.back());
        X.pop_back();
        y.pop_back();
                
        data_size--;
    }

    X_train = X;
    y_train = y;

    return {X_train, y_train, X_test, y_test};
}