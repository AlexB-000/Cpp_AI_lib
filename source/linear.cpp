#include "../include/deep_learn/linear.hpp"

Tensor Linear::forward(const Tensor& input){
    prevInput = input;
    return matmul(weights, input) + biases;
}

std::vector<Tensor> Linear::backward(const Tensor& prevDeriv){
    Tensor deriv = matmul(weights.T(), prevDeriv);
    Tensor d_biases = prevDeriv;
    Tensor d_weights = (prevInput * prevDeriv);
    return {d_weights, d_biases, deriv};
}

std::vector<Tensor*> Linear::get_parameters(){
    return {&weights, &biases};
}