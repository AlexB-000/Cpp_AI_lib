#include "../include/tensorMath.hpp"

Tensor matmul(const Tensor& t1, const Tensor& t2){
    if (t1.dim() > 2 || t2.dim() >2){
        throw std::invalid_argument("Multiplication of tensors only allowed for vectors and matrices");
    }
    if (t1.dim() == 2 && t2.dim() == 2){
        if (t1.shape != t2.T().shape){
            throw std::invalid_argument("Incompatible shape for matrix multiplication");
        }
        Tensor result {std::vector{t1.shape[0], t1.shape[0]}, 0};
        for (unsigned int i = 0; i<t1.shape[0]; i++){
            for (unsigned int j = 0; j<t1.shape[0]; j++){
                result[{i, j}] = 0;
                for (unsigned int k = 0; k<t1.shape[1]; k++){
                    result[{i, j}] = result[{i, j}] + t1[{i, k}] * t2[{k, j}];
                }
            }
        }
        return result;

    }
    if (t1.dim() == 1 && t2.dim() == 1){
        if (t1.shape != t2.T().shape){
            throw std::invalid_argument("Incompatible shape for vector multiplication");
        }
        Tensor result = 0;
        for (unsigned int i = 0; i<t1.shape[0]; i++){
            result = result + t1[i] * t2[i];
        }
        return result;
    }
    if (t1.dim() == 2 && t2.dim() == 1){
        if (t1.shape[1] != t2.shape[0]){
            throw std::invalid_argument("Incompatible shape for matrix-vector multiplication");
        }
        Tensor result {std::vector{t1.shape[0]}, 0};
        for (unsigned int i = 0; i<t1.shape[0]; i++){
            for (unsigned int j = 0; j<t1.shape[1]; j++){
                result[i] += t1[{i, j}] * t2[j];
            }
        }
        return result;
    }
    if (t1.shape[0] != t2.shape[0]){
        throw std::invalid_argument("Incompatible shape for vector-matrix multiplication");
    }
    Tensor result {std::vector{t2.shape[1]}, 0};
    for (unsigned int j = 0; j<t2.shape[1]; j++){
        for (unsigned int i = 0; i<t1.shape[0]; i++){
            result[j] += t1[i] * t2[{i, j}];
        }
    }
    return result;
}