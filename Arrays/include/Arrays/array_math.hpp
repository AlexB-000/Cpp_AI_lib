#pragma once
#include "array.hpp"

template<typename T>
Array<T> matrix_mult(const Array<T>& t1, const Array<T>& t2){
    if (t1.shape[1] != t2.shape[0]) {
        throw std::invalid_argument("Number of columns in the first matrix must match the number of rows in the second matrix.");
    }

    const uint32_t m = t1.shape[0];
    const uint32_t n = t1.shape[1];
    const uint32_t p = t2.shape[1];

    Array<T> result(std::vector<uint32_t>{m, p});

    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t k = 0; k < n; ++k) {
            for (uint32_t j = 0; j < p; ++j) {
                (*result.data_ptr)[i * p + j] += (*t1.data_ptr)[i * t1.strides[0] + k * t1.strides[1]] *
                                                 (*t2.data_ptr)[k * t2.strides[0] + j * t2.strides[1]];
            }
        }
    }
    return result;
}

template<typename T>
T vector_mult(const Array<T>& t1, const Array<T>& t2){
    if (t1.shape != t2.shape) {
        throw std::invalid_argument("vectors must have the same shape for dot product");
    }

    T sum = T();
    for (uint32_t i = 0; i < t1.data_ptr->size(); ++i) {
        sum += (*t1.data_ptr)[i * t1.strides[0]] * (*t2.data_ptr)[i * t2.strides[0]];
    }
    return sum;
}
template<typename T>
Array<T> matrix_vector_mult(const Array<T>& t1, const Array<T>& t2){
    if (t1.shape[1] != t2.shape[0]) {
        throw std::invalid_argument("Number of columns in the matrix must match the length of the vector.");
    }

    const uint32_t m = t1.shape[0];
    const uint32_t n = t1.shape[1];

    Array<T> result(std::vector<uint32_t>{m}, 0.0f);

    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            (*result.data_ptr)[i] += (*t1.data_ptr)[i * t1.strides[0] + j * t1.strides[1]] * (*t2.data_ptr)[j * t2.strides[0]];
        }
    }
    return result;
}

template<typename T>
Array<T> vector_matrix_mult(const Array<T>& t1, const Array<T>& t2){
    if (t1.shape[0] != t2.shape[0]) {
        throw std::invalid_argument("Length of the vector must match the number of lines in the matrix.");
    }
    const uint32_t m = t2.shape[0];
    const uint32_t n = t2.shape[1];
    
    Array<T> result(std::vector<uint32_t>{n}, 0.0f);
    
    for (uint32_t j = 0; j < m; ++j) {
        for (uint32_t k = 0; k < n; ++k) {
            (*result.data_ptr)[k] += (*t1.data_ptr)[j * t1.strides[0]] * (*t2.data_ptr)[j * t2.strides[0] + k * t2.strides[1]];
        }
    }
    return result;
}

/*
    Implement dot product.
        - For 2D arrays, perform standard matrix multiplication.
        - For 1D and 2D arrays, treat the 1D array as a row or column vector and perform the appropriate multiplication.
        - For 1D arrays, perform dot product.
        - More than 2D arrays ca
*/
template<typename T>
Array<T> dot(const Array<T>& t1, const Array<T>& t2){
    if (t1.dim == 2 && t2.dim == 2) {
        return matrix_mult(t1, t2);
    }
    else if (t1.dim == 2 && t2.dim == 1){
        return matrix_vector_mult(t1, t2);
    }
    else if (t1.dim == 1 && t2.dim == 2){
        return vector_matrix_mult(t1, t2);
    }
    else if (t1.dim == 1 && t2.dim == 1){
        return vector_mult(t1, t2);
    }
    else {
        throw std::invalid_argument("Only 1D and 2D arrays are supported for dot product.");
    }
}

//MARK: TODO
/*
    Implement matrix multiplication.
        - For 2D arrays, perform standard matrix multiplication.
        - 1D array treated as a row or column vector.
        - More than 2D arrays treated as stacks of 2D arrays.
*/
template<typename T>
Array<T> matmul(const Array<T>& t1, const Array<T>& t2){
    if (t1.dim == 2 && t2.dim == 2) {
        return matrix_mult(t1, t2);
    }
    else if (t1.dim == 2 && t2.dim == 1){
        return matrix_vector_mult(t1, t2);
    }
    else if (t1.dim == 1 && t2.dim == 2){
        return vector_matrix_mult(t1, t2);
    }
    else if (t1.dim == 1 && t2.dim == 1){
        return vector_mult(t1, t2);
    }
    else {
        throw std::invalid_argument("Only 1D and 2D arrays are supported for matmul.");
    }
    // std::vector<uint32_t> strides1 = t1.strides, strides2 = t2.strides;
    // if (t1.dim == 1) {
    //     strides1.insert(strides1.begin(), t1.shape[0]);
    // }
    // if (t2.dim == 1) {
    //     strides2.push_back(t2.shape[0]);
    // }

    // if (t1.shape.back() != *(t2.shape.end() - 2)) {
    //     throw std::invalid_argument("Dimension mismatch: number of columns in left matrix must match the number of rows in the right matrix.");
    // }
    // for (size_t i = 0; i < t1.shape.size() - 1; ++i) {
    //     if (t1.shape[i] != t2.shape[i]) {
    //         throw std::invalid_argument("Dimension mismatch: shapes are not compatible for broadcasting.");
    //     }
    // }

    // std::vector<uint32_t> result_shape = t1.shape;
    // result_shape.back() = t2.shape.back();
    // Array<T> result(result_shape);

    // std::vector<uint32_t> idx(result.dim, 0);
    // while (true){
    //     uint32_t flat_index1 = 0, flat_index2 = 0;
    //     for (size_t i = 0; i < bshape.size(); ++i) {
    //         flat_index1 += idx[i] * bstrides1[i];
    //         flat_index2 += idx[i] * bstrides2[i];
    //     }
    //     (*result.data_ptr)[flat_index1] = (*t1.data_ptr)[flat_index1] op_sign (*t2.data_ptr)[flat_index2];
    //     /* Increment the multi-dimensional index */
    //     for (uint32_t i = bshape.size() - 1; i >= 0; --i) {
    //         if (++idx[i] < bshape[i]) {
    //             break;
    //         }
    //         idx[i] = 0;
    //         if (i == 0) {
    //             return result;
    //         }
    //     }
    // }
}