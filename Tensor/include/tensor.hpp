#pragma once
#include <iostream>
#include <stdexcept>
#include <variant>
#include <vector>
#include <memory>
#include <algorithm>
#include "slice.hpp"

class Tensor {
    // main full constructor
    Tensor(const std::vector<unsigned int> inShape, const std::vector<unsigned int> inStrides,
    const std::shared_ptr<std::vector<float>>& inData, const unsigned int inOffset, bool is_subtensor);

    inline void _create_new_tensor(const std::vector<unsigned int>& shape, const float value=0);
    inline void _deep_copy(const Tensor& other);
public:
    // main constructors
    Tensor(const std::vector<unsigned int>& shape, const float value=0);
    Tensor(const Tensor& other);

    // other constructors
    Tensor(const std::vector<unsigned int>& shape, const std::vector<float>& initData) :
        Tensor(shape) {
            data = std::make_shared<std::vector<float>>(initData);
    }
    Tensor(const float value=0) : Tensor(std::vector<unsigned int>{}, value) {};

    Tensor& operator=(const Tensor& other);

    Tensor copy() const ;

    void show(unsigned int indent=0) const;

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

    friend Tensor fast_elementwise_add(const Tensor& A, const Tensor& B);
    friend Tensor fast_elementwise_sub(const Tensor& A, const Tensor& B);
    friend Tensor fast_elementwise_mul(const Tensor& A, const Tensor& B);
    friend Tensor fast_elementwise_div(const Tensor& A, const Tensor& B);

    std::shared_ptr<std::vector<float>> data;

    bool _is_subtensor = false;
    unsigned int offset = 0; // Offset for views

    std::vector <unsigned int> strides;
    std::vector <unsigned int> shape;
};

// MARK: friend functions

Tensor cat(const std::vector< Tensor >& tensors, unsigned int axis=0);
Tensor stack(const std::vector< Tensor >& tensors, unsigned int axis=0);

Tensor fast_elementwise_add(const Tensor& A, const Tensor& B);
Tensor fast_elementwise_sub(const Tensor& A, const Tensor& B);
Tensor fast_elementwise_mul(const Tensor& A, const Tensor& B);
Tensor fast_elementwise_div(const Tensor& A, const Tensor& B);