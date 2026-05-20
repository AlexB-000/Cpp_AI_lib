#pragma once
#include "array.hpp"

template<typename T>
void broadcast(const Array<T>& array1, const Array<T>& array2,
    std::vector<uint32_t>& bshape,
    std::vector<uint32_t>& bstrides1, std::vector<uint32_t>& bstrides2) {

    uint32_t result_dim, excess1 = 0, excess2 = 0;
    if (array1.dim > array2.dim) {
        result_dim = array1.dim;
        excess2 = result_dim - array2.dim;
    }
    else {
        result_dim = array2.dim;
        excess1 = result_dim - array1.dim;
    }
    bshape.resize(result_dim); bstrides1.resize(result_dim); bstrides2.resize(result_dim);

    int32_t i1 = array1.dim - 1, i2 = array2.dim - 1;
    int32_t ib = result_dim - 1; 

    while (i1 >= 0 && i2 >= 0) {
        if (array1.shape[i1] == array2.shape[i2]) {
            bshape[ib] = array1.shape[i1];
            bstrides1[ib] = array1.strides[i1];
            bstrides2[ib] = array2.strides[i2];
        }
        else if (array1.shape[i1] == 1) {
            bshape[ib] = array2.shape[i2];
            bstrides1[ib] = 0;
            bstrides2[ib] = array2.strides[i2];
        }
        else if (array2.shape[i2] == 1) {
            bshape[ib] = array1.shape[i1];
            bstrides1[ib] = array1.strides[i1];
            bstrides2[ib] = 0;
        }
        else {
            throw std::invalid_argument("Could not broadcast shapes: " + std::to_string(array1.shape[i1]) + " and " + std::to_string(array2.shape[i2]));
        }
        --i1; --i2; --ib;
    }
    while (i1 >= 0) {
        bshape[ib] = array1.shape[i1];
        bstrides1[ib] = array1.strides[i1];
        bstrides2[ib] = 0;
        --i1; --ib;
    }
    while (i2 >= 0) {
        bshape[ib] = array2.shape[i2];
        bstrides1[ib] = 0;
        bstrides2[ib] = array2.strides[i2];
        --i2; --ib;
    }
    return;
}

#define defineElementwiseOp(op, op_sign) \
template<typename T> \
Array<T> op(const Array<T>& t1, const Array<T>& t2) { \
    std::vector<uint32_t> bshape, bstrides1, bstrides2; \
    broadcast(t1, t2, bshape, bstrides1, bstrides2); \
    Array<T> result(bshape); \
    uint32_t total_size = 1; \
    for (uint32_t dim : bshape) { \
        total_size *= dim; \
    } \
    std::vector<uint32_t> idx(bshape.size(), 0); \
    while (true){ \
        uint32_t flat_index1 = 0, flat_index2 = 0; \
        for (size_t i = 0; i < bshape.size(); ++i) { \
            flat_index1 += idx[i] * bstrides1[i]; \
            flat_index2 += idx[i] * bstrides2[i]; \
        } \
        result.data[flat_index1] = t1.data[flat_index1] op_sign t2.data[flat_index2]; \
        /* Increment the multi-dimensional index */ \
        for (uint32_t i = bshape.size() - 1; i >= 0; --i) { \
            if (++idx[i] < bshape[i]) { \
                break; \
            } \
            idx[i] = 0; \
            if (i == 0) { \
                return result; \
            } \
        } \
    } \
    return result; \
} \
template<typename T> \
inline Array<T> op(const Array<T>& t, const T scalar){ \
    return t op_sign Array<T>(scalar); \
}

#define defineElementwiseOpInPlace(op, op_sign) \
template<typename T> \
Array<T>& op(Array<T>& t1, const Array<T>& t2) { \
    std::vector<uint32_t> bshape, bstrides1, bstrides2; \
    broadcast(t1, t2, bshape, bstrides1, bstrides2); \
    if (bshape != t1.shape) { \
        throw std::invalid_argument("Dimension mismatch: cannot broadcast shape to add to the array."); \
    } \
    uint32_t total_size = 1; \
    for (uint32_t dim : bshape) { \
        total_size *= dim; \
    } \
    std::vector<uint32_t> idx(bshape.size(), 0); \
    while (true){ \
        uint32_t flat_index1 = 0, flat_index2 = 0; \
        for (size_t i = 0; i < bshape.size(); ++i) { \
            flat_index1 += idx[i] * bstrides1[i]; \
            flat_index2 += idx[i] * bstrides2[i]; \
        } \
        t1.data[flat_index1] op_sign t2.data[flat_index2]; \
        /* Increment the multi-dimensional index */ \
        for (uint32_t i = bshape.size() - 1; i >= 0; --i) { \
            if (++idx[i] < bshape[i]) { \
                break; \
            } \
            idx[i] = 0; \
            if (i == 0) { \
                return t1; \
            } \
        } \
    } \
    return t1; \
} \
template<typename T> \
inline Array<T>& op(Array<T>& t, const T scalar){ \
    return t op_sign Array<T>(scalar); \
}

defineElementwiseOp(operator+, +)
defineElementwiseOp(operator-, -)
defineElementwiseOp(operator*, *)
defineElementwiseOp(operator/, /)

defineElementwiseOpInPlace(operator+=, +=)
defineElementwiseOpInPlace(operator-=, -=)
defineElementwiseOpInPlace(operator*=, *=)
defineElementwiseOpInPlace(operator/=, /=)

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
                result.data[i * p + j] += t1.data[i * n + k] * t2.data[k * p + j];
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
    for (uint32_t i = 0; i < t1.data.size(); ++i) {
        sum += t1.data[i] * t2.data[i];
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
            result.data[i] += t1.data[i * n + j] * t2.data[j];
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
            result.data[k] += t1.data[j] * t2.data[j * n + k];
        }
    }
    return result;
}

//MARK: TODO
/*
    Implement matrix multiplication.
        - For 2D arrays, perform standard matrix multiplication.
        - For 1D and 2D arrays, treat the 1D array as a row or column vector and perform the appropriate multiplication.
        - For 1D arrays, perform dot product.
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
    //     result.data[flat_index1] = t1.data[flat_index1] op_sign t2.data[flat_index2];
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