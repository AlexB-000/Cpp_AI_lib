#pragma once
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <algorithm>

template<typename _T>
struct Array {
    uint32_t dim;
    std::vector<uint32_t> shape;
    std::vector<_T> data;

    Array() = default;
    
    Array(const Array& other) : shape(other.shape), data(other.data), dim(other.dim) {}
    Array(const _T& value): shape({}), dim(0), data({value}) {}

    Array(const std::vector<uint32_t>& shape, const _T& value=_T()) : shape(shape), dim(shape.size()) {
        uint32_t total_size = 1;
        for (uint32_t dim : shape) {
            total_size *= dim;
        }
        data.resize(total_size, value);
    }
    Array(const std::vector<uint32_t>& shape, const std::vector<_T>& data) :
        shape(shape), dim(shape.size()), data(data) {}

    Array& operator=(const Array& other) {
        shape = other.shape;
        data = other.data;
        dim = other.dim;
        return *this;
    }

    Array T() const {
        if (dim != 2) {
            throw std::invalid_argument("Transpose is only defined for 2D arrays.");
        }
        uint32_t rows = shape[0];
        uint32_t cols = shape[1];
        Array result({cols, rows});
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                result.data[j * rows + i] = data[i * cols + j];
            }
        }
        return result;
    }

    _T operator[](uint32_t index) const {
        return (*this)[std::vector<uint32_t>{index}];
    }
    _T operator[](const std::vector<uint32_t>& indices) const {
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("Number of indices must match the number of dim.");
        }
        uint32_t flat_index = 0;
        uint32_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds.");
            }
            flat_index += indices[i] * stride;
            stride *= shape[i];
        }
        return data[flat_index];
    }

    _T& operator[](uint32_t index) {
        return (*this)[std::vector<uint32_t>{index}];
    }

    _T& operator[](const std::vector<uint32_t>& indices) {
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("Number of indices must match the number of dim.");
        }
        uint32_t flat_index = 0;
        uint32_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds.");
            }
            flat_index += indices[i] * stride;
            stride *= shape[i];
        }
        return data[flat_index];
    }

    //MARK: element-wise
    Array operator+(const Array& other) const {
        Array result(shape);
        if (shape != other.shape) {
            throw std::invalid_argument("Shapes must match for addition.");
        }
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }
    Array operator-(const Array& other) const {
        Array result(shape);
        if (shape != other.shape) {
            throw std::invalid_argument("Shapes must match for subtraction.");
        }
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }
    Array operator*(const Array& other) const {
        Array result(shape);
        if (shape != other.shape) {
            throw std::invalid_argument("Shapes must match for multiplication.");
        }
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }
    Array operator/(const Array& other) const {
        Array result(shape);
        if (shape != other.shape) {
            throw std::invalid_argument("Shapes must match for division.");
        }
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] / other.data[i];
        }
        return result;
    }

    Array& operator+=(const Array& other) {
        if (shape != other.shape) {
            throw std::invalid_argument("Shapes must match for addition.");
        }
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += other.data[i];
        }
        return *this;
    }
    Array& operator-=(const Array& other) {
        if (shape != other.shape) {
            throw std::invalid_argument("Shapes must match for subtraction.");
        }
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] -= other.data[i];
        }
        return *this;
    }
    Array& operator*=(const Array& other) {
        if (shape != other.shape) {
            throw std::invalid_argument("Shapes must match for multiplication.");
        }
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] *= other.data[i];
        }
        return *this;
    }
    Array& operator/=(const Array& other) {
        if (shape != other.shape) {
            throw std::invalid_argument("Shapes must match for division.");
        }
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] /= other.data[i];
        }
        return *this;
    }

    //MARK: scalar
    Array operator+(const _T& scalar) const {
        Array result(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + scalar;
        }
        return result;
    }
    Array operator-(const _T& scalar) const {
        Array result(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] - scalar;
        }
        return result;
    }
    Array operator*(const _T& scalar) const {
        Array result(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }
    Array operator/(const _T& scalar) const {
        Array result(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] / scalar;
        }
        return result;
    }

    Array& operator+=(const _T& scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += scalar;
        }
        return *this;
    }
    Array& operator-=(const _T& scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] -= scalar;
        }
        return *this;
    }
    Array& operator*=(const _T& scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] *= scalar;
        }
        return *this;
    }
    Array& operator/=(const _T& scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] /= scalar;
        }
        return *this;
    }

    //MARK: show
    void show() const {
        if (dim == 0) {
            std::cout << data[0] << std::endl;
            return;
        }
        std::cout << "[";
        if (dim == 1) {
            for (size_t i = 0; i < data.size(); ++i) {
                std::cout << data[i];
                if (i < data.size() - 1) {
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
                    std::cout << data[i * cols + j];
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
            for (size_t i = 0; i < data.size(); ++i) {
                std::cout << data[i];
                if (i < data.size() - 1) {
                    std::cout << ", ";
                }
            }
        }
        std::cout << "]" << std::endl;
    }
};