#include "../../include/deep_learning/linear.hpp"

Tensor Linear::forward(const Tensor& input){
    prevInput = input;
    return matmul(weights, input) + biases;
}

std::vector<Tensor> Linear::backward(const Tensor& prevDeriv){
    Tensor deriv = matmul(weights.T(), prevDeriv);
    Tensor d_biases = prevDeriv;
    Tensor d_weights {std::vector<unsigned int>{static_cast<unsigned int>(outputSize), static_cast<unsigned int>(inputSize)}, 0};
    for (unsigned int i=0; i<static_cast<unsigned int>(outputSize); i++){
        for (unsigned int j=0; j<static_cast<unsigned int>(inputSize); j++){
            d_weights[{i, j}] = prevDeriv[i] * prevInput[j];
        }
    }
    return {d_weights, d_biases, deriv};
}

std::vector<Tensor*> Linear::get_parameters(){
    return {&weights, &biases};
}