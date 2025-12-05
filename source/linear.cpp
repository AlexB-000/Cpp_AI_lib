#include "linear.hpp"

Tensor<double> Linear::forward(const Tensor<double>& input){
    prevInput = input;
    return matmul(weights, input) + biases;
}

std::vector<Tensor<double>> Linear::backward(const Tensor<double>& prevDeriv){
    Tensor<double> deriv = matmul(weights.T(), prevDeriv);
    Tensor<double> d_biases = prevDeriv;
    Tensor<double> d_weights = (prevInput * prevDeriv);
    return {d_weights, d_biases, deriv};
}