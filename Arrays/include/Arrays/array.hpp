#pragma once
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <algorithm>

template<typename _T>
struct Array {
    uint32_t dim;
    uint32_t offset;
    std::vector<uint32_t> shape;
    std::vector<uint32_t> strides;
    std::shared_ptr<std::vector<_T>> data_ptr;
private:
    Array();
public:
    Array(const Array& other) : dim(other.dim), offset(other.offset), shape(other.shape), strides(other.strides),
        data_ptr(std::make_shared<std::vector<_T>>(*other.data_ptr)) {}
    Array(const _T& value): shape({}), strides({}), dim(0), offset(0),
        data_ptr(std::make_shared<std::vector<_T>>(std::vector<_T>{value})) {}

    Array(const std::vector<uint32_t>& shape, const _T& value=_T()) : shape(shape), strides(dim), dim(shape.size()), offset(0) {
        uint32_t size = 1;
        for (int32_t i = dim-1; i >= 0; --i) {
            strides[i] = size;
            size *= shape[i];
        }
        data_ptr = std::make_shared<std::vector<_T>>(std::vector<_T>(size, value));
    }
    Array(const std::vector<uint32_t>& shape, const std::vector<_T>& data) :
        shape(shape), strides(dim), dim(shape.size()), offset(0), data_ptr(std::make_shared<std::vector<_T>>(data)) {
        uint32_t size = 1;
        for (int32_t i = dim-1; i >= 0; --i) {
            strides[i] = size;
            size *= shape[i];
        }
    }

    #define defineElementwiseOp(op, op_sign) \
    Array<_T> op(const Array<_T>& other) const { \
        std::vector<uint32_t> bshape, bstrides1, bstrides2; \
        broadcast(*this, other, bshape, bstrides1, bstrides2); \
        Array<_T> result(bshape); \
        std::vector<uint32_t> idx(bshape.size(), 0); \
        while (true){ \
            uint32_t flat_indexR = 0, flat_index1 = offset, flat_index2 = other.offset; \
            for (size_t i = 0; i < bshape.size(); ++i) { \
                flat_indexR += idx[i] * result.strides[i]; \
                flat_index1 += idx[i] * bstrides1[i]; \
                flat_index2 += idx[i] * bstrides2[i]; \
            } \
            (*result.data_ptr)[flat_indexR] = (*data_ptr)[flat_index1] op_sign (*other.data_ptr)[flat_index2]; \
            /* Increment the multi-dimensional index */ \
            for (int32_t i = bshape.size() - 1; i >= 0; --i) { \
                if (++idx[i] < bshape[i]) { \
                    break; \
                } \
                idx[i] = 0; \
                if (i == 0) { \
                    return result; \
                } \
            } \
        } \
    } \
    inline Array<_T> op(const _T scalar) const { \
        return op(Array<_T>(scalar)); \
    }

    #define defineElementwiseOpInPlace(op, op_sign) \
    Array<_T>& op(const Array<_T>& other) { \
        std::vector<uint32_t> bshape, bstrides1, bstrides2; \
        broadcast(*this, other, bshape, bstrides1, bstrides2); \
        if (bshape != shape) { \
            throw std::invalid_argument("Dimension mismatch: cannot broadcast shape of the input to the array."); \
        } \
        std::vector<uint32_t> idx(bshape.size(), 0); \
        while (true){ \
            uint32_t flat_index1 = offset, flat_index2 = other.offset; \
            for (size_t i = 0; i < bshape.size(); ++i) { \
                flat_index1 += idx[i] * bstrides1[i]; \
                flat_index2 += idx[i] * bstrides2[i]; \
            } \
            (*data_ptr)[flat_index1] op_sign (*other.data_ptr)[flat_index2]; \
            /* Increment the multi-dimensional index */ \
            for (uint32_t i = bshape.size() - 1; i >= 0; --i) { \
                if (++idx[i] < bshape[i]) { \
                    break; \
                } \
                idx[i] = 0; \
                if (i == 0) { \
                    return *this; \
                } \
            } \
        } \
    } \
    inline Array<_T>& op(const _T scalar){ \
        return op(Array<_T>(scalar)); \
    }
    
    defineElementwiseOp(operator+, +)
    defineElementwiseOp(operator-, -)
    defineElementwiseOp(operator*, *)
    defineElementwiseOp(operator/, /)

    defineElementwiseOpInPlace(operator=, =)
    defineElementwiseOpInPlace(operator+=, +=)
    defineElementwiseOpInPlace(operator-=, -=)
    defineElementwiseOpInPlace(operator*=, *=)
    defineElementwiseOpInPlace(operator/=, /=)

    Array<_T> T() const {
        if (dim != 2) {
            throw std::invalid_argument("Transpose is only defined for 2D arrays.");
        }
        uint32_t rows = shape[0];
        uint32_t cols = shape[1];
        Array<_T> result({cols, rows});
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                (*result.data_ptr)[j * rows + i] = (*data_ptr)[i * strides[0] + j * strides[1] + offset];
            }
        }
        return result;
    }

    //MARK: indexing
    _T operator[](uint32_t index) const {
        return (*this)[std::vector<uint32_t>{index}];
    }
    _T operator[](const std::vector<uint32_t>& indices) const {
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("Number of indices must match the number of dim.");
        }
        uint32_t flat_index = offset;
        for (uint32_t i = 0; i < shape.size(); ++i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds.");
            }
            flat_index += indices[i] * strides[i];
        }
        return (*data_ptr)[flat_index];
    }

    _T& operator[](uint32_t index) {
        return (*this)[std::vector<uint32_t>{index}];
    }

    _T& operator[](const std::vector<uint32_t>& indices) {
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("Number of indices must match the number of dim.");
        }
        uint32_t flat_index = offset;
        for (uint32_t i = 0; i < shape.size(); ++i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds.");
            }
            flat_index += indices[i] * strides[i];
        }
        return (*data_ptr)[flat_index];
    }

    //MARK: show
    void show() const {
        if (dim == 0) {
            std::cout << (*data_ptr)[offset] << std::endl;
            return;
        }
        std::cout << "[";
        if (dim == 1) {
            for (size_t i = 0; i < shape[0]; ++i) {
                std::cout << (*data_ptr)[i * strides[0] + offset];
                if (i < shape[0] - 1) {
                    std::cout << ", ";
                }
            }
        }
        else if (dim == 2){
            uint32_t rows = shape[0];
            uint32_t cols = shape[1];
            for (uint32_t i = 0; i < rows; ++i) {
                std::cout << "[";
                for (uint32_t j = 0; j < cols; ++j) {
                    std::cout << (*data_ptr)[i * strides[0] + j * strides[1] + offset];
                    if (j < cols - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]";
                if (i < rows - 1) {
                    std::cout << ",\n";
                }
            }
        } else {
            std::cout << "Nd Array with dim=" << dim << " and shape=[";
            for (size_t i = 0; i < shape.size(); ++i) {
                std::cout << shape[i];
                if (i < shape.size() - 1) {
                    std::cout << ", ";
                }
            }
        }
        std::cout << "]" << std::endl;
    }
};

template<typename _T>
void broadcast(const Array<_T>& array1, const Array<_T>& array2,
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