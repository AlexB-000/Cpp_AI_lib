#pragma once
#include <vector>
#include "../../Tensor/include/tensorMath.hpp"

std::vector<std::vector<Tensor>> train_test_split(std::vector<Tensor> X, std::vector<Tensor> y, float test_prop=0.1){
    std::vector<Tensor> X_train, y_train, X_test, y_test;
    
    //random numbers generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
    uint data_size = X.size();
    uint test_size = X.size() * test_prop;
    for (uint i = 0; i<test_size; i++){
        std::uniform_int_distribution<> dis(0, data_size);
        uint idx = dis(gen);
        X_test.push_back(X[idx]);
        y_test.push_back(y[idx]);

        X.erase(X.begin() + idx);
        y.erase(y.begin() + idx);
                
        data_size--;
    }

    X_train = X;
    y_train = y;

    return {X_train, y_train, X_test, y_test};
}