#include "CppAI/deep_learning/linear.hpp"

Array<float> Linear::forward(const Array<float>& input){
    if (input.dim != 1){
        throw std::invalid_argument("In Linear forward : Input must be a 1D array.");
    }
    if (input.shape[0] != inputSize) {
        throw std::invalid_argument("In Linear forward : Input size does not match the expected size.");
    }
    if(training) prevInput = input;
    Array<float> output(nd::matrix_vector_mult(weights, input));
    for (uint32_t i=0; i<output.shape[0]; ++i){
        (*output.data_ptr)[i] += (*biases.data_ptr)[i];
    }
    return output;
}

std::vector<Array<float>> Linear::backward(const Array<float>& prevDeriv){
    if (prevDeriv.dim != 1){
        throw std::invalid_argument("In Linear backward : Previous derivative must be a 1D array.");
    }
    if (prevDeriv.shape[0] != outputSize) {
        throw std::invalid_argument("In Linear backward : Previous derivative size does not match the expected output size.");
    }
    Array<float> deriv = nd::vector_matrix_mult(prevDeriv, weights);
    Array<float> d_biases = prevDeriv;
    Array<float> d_weights {std::vector<unsigned int>{outputSize, inputSize}, 0};
    for (unsigned int i=0; i<static_cast<unsigned int>(outputSize); i++){
        for (unsigned int j=0; j<static_cast<unsigned int>(inputSize); j++){
            (*d_weights.data_ptr)[i*d_weights.strides[0] + j] =
                (*prevDeriv.data_ptr)[i*prevDeriv.strides[0] + prevDeriv.offset]
                * (*prevInput.data_ptr)[j];
        }
    }
    return {d_weights, d_biases, deriv};
}