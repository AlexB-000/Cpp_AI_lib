#pragma once
#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <vector>
#include <memory>
#include <algorithm>

template<typename _T>
struct Array {
    uint32_t dim;
    uint32_t offset;
    bool owner;
    std::vector<uint32_t> shape;
    std::vector<uint32_t> strides;
    std::shared_ptr<std::vector<_T>> data_ptr;
private:
    Array(uint32_t inDim, uint32_t inOffset, bool owner, const std::vector<uint32_t>& inShape, const std::vector<uint32_t>& inStrides,
        std::shared_ptr<std::vector<_T>> inDataPtr) :
        dim(inDim), offset(inOffset), owner(owner), shape(inShape), strides(inStrides), data_ptr(inDataPtr){};
public:
    Array(const Array& other) : dim(other.dim), offset(0), shape(other.shape), strides(other.dim) {
        if (!other.owner) {
            offset = other.offset;
            strides = other.strides;
            data_ptr = other.data_ptr;
            owner = false;
            return;
        }
        owner = true;
        uint32_t size = 1;
        for (int32_t i = dim-1; i >= 0; --i) {
            strides[i] = size;
            size *= shape[i];
        }
        data_ptr = std::make_shared<std::vector<_T>>(std::vector<_T>(size));
        *this = other;
    }
    Array(const _T& value): shape({}), strides({}), dim(0), offset(0), owner(true),
        data_ptr(std::make_shared<std::vector<_T>>(std::vector<_T>{value})) {}

    Array(const std::vector<uint32_t>& shape, const _T& value=_T()) : shape(shape), strides(dim),
        dim(shape.size()), offset(0), owner(true) {
        uint32_t size = 1;
        for (int32_t i = dim-1; i >= 0; --i) {
            strides[i] = size;
            size *= shape[i];
        }
        data_ptr = std::make_shared<std::vector<_T>>(std::vector<_T>(size, value));
    }
    Array(const std::vector<uint32_t>& shape, const std::vector<_T>& data) :
        shape(shape), strides(dim), dim(shape.size()), offset(0), owner(true),
        data_ptr(std::make_shared<std::vector<_T>>(data)) {
        uint32_t size = 1;
        for (int32_t i = dim-1; i >= 0; --i) {
            strides[i] = size;
            size *= shape[i];
        }
    }

    Array<_T> copy() const {
        Array<_T> result (shape);
        result = *this;
        return result;
    }

    #define defineElementwiseOp(op, op_sign, return_type) \
    Array<return_type> op(const Array<_T>& other) const { \
        if (owner && other.owner && shape==other.shape){ \
            Array<return_type> result(shape); \
            /* operands aren't views and have the same shape so we optimize and just go through the memory */ \
            for (uint32_t i=0; i<data_ptr->size(); ++i) \
                (*result.data_ptr)[i] = (*data_ptr)[i] op_sign (*other.data_ptr)[i]; \
            return result; \
        } \
        std::vector<uint32_t> bshape, bstrides1, bstrides2; \
        broadcast(*this, other, bshape, bstrides1, bstrides2); \
        Array<return_type> result(bshape); \
        if (bshape.size() == 0) { \
            (*result.data_ptr)[0] = (*data_ptr)[offset] op_sign (*other.data_ptr)[other.offset]; \
            return result; \
        } \
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
    inline Array<return_type> op(const _T scalar) const { \
        return op(Array<return_type>(scalar)); \
    }

    #define defineElementwiseOpInPlace(op, op_sign) \
    Array<_T>& op(const Array<_T>& other) { \
        if (owner && other.owner && shape==other.shape){ \
            /* operands aren't views and have the same shape so we optimize and just go through the memory */ \
            for (uint32_t i=0; i<data_ptr->size(); ++i) \
                (*data_ptr)[i] op_sign (*other.data_ptr)[i]; \
            return *this; \
        } \
        std::vector<uint32_t> bshape, bstrides1, bstrides2; \
        broadcast(*this, other, bshape, bstrides1, bstrides2); \
        if (bshape != shape) { \
            throw std::invalid_argument("Dimension mismatch: cannot broadcast shape of the input to the array."); \
        } \
        if (bshape.size() == 0) { \
            (*data_ptr)[offset] op_sign (*other.data_ptr)[other.offset]; \
            return *this; \
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
            for (int32_t i = bshape.size() - 1; i >= 0; --i) { \
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

    defineElementwiseOp(operator+, +, _T)
    defineElementwiseOp(operator-, -, _T)
    defineElementwiseOp(operator*, *, _T)
    defineElementwiseOp(operator/, /, _T)

    defineElementwiseOp(operator==, ==, bool)
    defineElementwiseOp(operator!=, !=, bool)
    defineElementwiseOp(operator<, <, bool)
    defineElementwiseOp(operator<=, <=, bool)
    defineElementwiseOp(operator>, >, bool)
    defineElementwiseOp(operator>=, >=, bool)

    defineElementwiseOpInPlace(operator=, =)
    defineElementwiseOpInPlace(operator+=, +=)
    defineElementwiseOpInPlace(operator-=, -=)
    defineElementwiseOpInPlace(operator*=, *=)
    defineElementwiseOpInPlace(operator/=, /=)

    Array<_T> operator-() const {
        Array<_T> result(shape);
        std::vector<uint32_t> idx(dim, 0);
        while (true){
            uint32_t  resultIDX = 0, arrIdx = offset;
            for (uint32_t i = 0; i < dim; ++i) {
                resultIDX += idx[i] * result.strides[i];
                arrIdx += idx[i] * strides[i];
            }
            (*result.data_ptr)[resultIDX] = -(*data_ptr)[arrIdx];
            /* Increment the multi-dimensional index */
            for (int32_t i = dim - 1; i >= 0; --i) {
                if (++idx[i] < shape[i]) {
                    break;
                }
                idx[i] = 0;
                if (i == 0) {
                    return result;
                }
            }
        }
    }

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
    Array<_T> operator[](const uint32_t index) const {
        if (dim == 0) {
            throw std::invalid_argument("Cannot index into a scalar.");
        }
        if (index >= shape[0]) {
            throw std::out_of_range("Index " + std::to_string(index) + " out of bounds for dimension 0 with size " + std::to_string(shape[0]));
        }
        uint32_t new_offset = offset + index * strides[0];
        std::vector<uint32_t> new_shape(shape.begin() + 1, shape.end());
        std::vector<uint32_t> new_strides(strides.begin() + 1, strides.end());
        return Array<_T>(new_shape.size(), new_offset, false, new_shape, new_strides, data_ptr);
    }

    Array<_T> operator[](const std::vector<uint32_t>& indices) const {
        if (indices.size() > shape.size()) {
            throw std::invalid_argument("Number of indices must not exceed the number of dimensions.");
        }
        std::vector<uint32_t> new_shape(shape.begin()+indices.size(), shape.end());
        std::vector<uint32_t> new_strides(strides.begin()+indices.size(), strides.end());
        uint32_t new_offset = offset;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index " + std::to_string(indices[i]) + " out of bounds for dimension " + std::to_string(i) + " with size " + std::to_string(shape[i]));
            }
            new_offset += indices[i] * strides[i];
        }
        return Array<_T>(new_shape.size(), new_offset, false, new_shape, new_strides, data_ptr);
    }

    _T at(uint32_t index) const {
        if (dim == 0) {
            return (*data_ptr)[offset];
        }
        uint32_t flat_index = offset;
        if (index >= shape[0]) {
            throw std::out_of_range("Index " + std::to_string(index) + " out of bounds for dimension 0 with size " + std::to_string(shape[0]));
        }
        flat_index += index * strides[0];
        return (*data_ptr)[flat_index];
    }
    _T at(const std::vector<uint32_t>& indices) const {
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("Number of indices must match the number of dimensions.");
        }
        uint32_t flat_index = offset;
        for (uint32_t i = 0; i < shape.size(); ++i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index " + std::to_string(indices[i]) + " out of bounds for dimension " + std::to_string(i) + " with size " + std::to_string(shape[i]));
            }
            flat_index += indices[i] * strides[i];
        }
        return (*data_ptr)[flat_index];
    }

    _T& at(uint32_t index) {
        if (dim == 0) {
            return (*data_ptr)[offset];
        }
        uint32_t flat_index = offset;
        if (index >= shape[0]) {
            throw std::out_of_range("Index " + std::to_string(index) + " out of bounds for dimension 0 with size " + std::to_string(shape[0]));
        }
        flat_index += index * strides[0];
        return (*data_ptr)[flat_index];
    }

    _T& at(const std::vector<uint32_t>& indices) {
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("Number of indices must match the number of dimensions.");
        }
        uint32_t flat_index = offset;
        for (uint32_t i = 0; i < shape.size(); ++i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index " + std::to_string(indices[i]) + " out of bounds for dimension " + std::to_string(i) + " with size " + std::to_string(shape[i]));
            }
            flat_index += indices[i] * strides[i];
        }
        return (*data_ptr)[flat_index];
    }

    //MARK: show
    void show(const std::string& shift = "") const {
        if (dim == 0) {
            std::cout << shift << (*data_ptr)[offset] << std::endl;
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
                if (i > 0) {
                    std::cout << " ";
                }
                std::cout << "[";
                for (uint32_t j = 0; j < cols; ++j) {
                    std::cout << (*data_ptr)[i * strides[0] + j * strides[1] + offset];
                    if (j < cols - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]";
                if (i < rows - 1) {
                    std::cout << ",\n" << shift;
                }
            }
        } else {
            for (uint32_t i = 0; i < shape[0]; ++i) {
                if (i > 0) {
                    std::cout << " ";
                }
                Array<_T> sub_array = (*this)[i];
                sub_array.show(shift + " ");
                if (i < shape[0] - 1) {
                    std::cout << ",\n";
                }
            }
        }
        std::cout << "]";
        if (shift.empty()) {
            std::cout << std::endl;
        }
    }
};

//MARK: broadcast
template<typename _T>
void broadcast(const Array<_T>& array1, const Array<_T>& array2,
    std::vector<uint32_t>& bshape,
    std::vector<uint32_t>& bstrides1, std::vector<uint32_t>& bstrides2) {

    uint32_t result_dim = array1.dim > array2.dim ? array1.dim : array2.dim;

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