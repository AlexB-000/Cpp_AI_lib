#pragma once
#include <iostream>
#include <stdexcept>
#include <variant>
#include <vector>
#include <memory>
#include <algorithm>
#include "slice.hpp"

class Tensor {
public:
    Tensor(const std::vector<unsigned int>& shape, const float value=0);
    Tensor(const float value=0) : Tensor(std::vector<unsigned int>{}, value) {};
    Tensor(const std::vector<unsigned int>& shape, const std::vector<float>& initData) :
        Tensor(shape) {
            data = std::make_shared<std::vector<float>>(initData);
    }
    Tensor(const std::vector<unsigned int> inShape, const std::vector<unsigned int> inStrides,
    const std::shared_ptr<std::vector<float>>& inData, const unsigned int inOffset);
    
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    Tensor copy() const ;

    void show(unsigned int indent=0);

    unsigned int dim() const { return shape.size(); };

    Tensor T() const;

    Tensor collapse(unsigned int axis=0) const;

    
    friend Tensor cat(const std::vector< Tensor >& tensors, unsigned int axis);
    
    friend Tensor stack(const std::vector< Tensor >& tensors, unsigned int axis);

    float& operator[](const std::vector<unsigned int>& indices) const;
    float& operator[](unsigned int index) const { return (*this)[std::vector<unsigned int>{index}]; }
    Tensor subtensor(unsigned int index) const;

    Tensor slice(const std::vector<slice>& slices);

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    std::shared_ptr<std::vector<float>> data;

    unsigned int offset = 0; // Offset for views/slices

    std::vector <unsigned int> strides;
    std::vector <unsigned int> shape;
};

// MARK: friend functions

Tensor cat(const std::vector< Tensor >& tensors, unsigned int axis=0);
Tensor stack(const std::vector< Tensor >& tensors, unsigned int axis=0);