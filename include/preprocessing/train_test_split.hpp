#pragma once
#include <vector>
#include "../../Tensor/include/tensorMath.hpp"

std::vector<std::vector<Tensor>> train_test_split(std::vector<Tensor> X, std::vector<Tensor> y, float test_prop=0.1){
    std::vector<Tensor> X_train, y_train, X_test, y_test;
    
    //random numbers generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
    uint test_size = X.size() * test_prop;
    for (uint i = 0; i<test_size; i++){
        uint data_size = X.size();
        std::uniform_int_distribution<> dis(0, data_size - 1);
        uint idx = dis(gen);
        X_test.push_back(X[idx]);
        y_test.push_back(y[idx]);

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