#pragma once
#include <iostream>
#include <stdexcept>
#include <variant>
#include <vector>
#include <memory>
#include <algorithm>
#include "slice.hpp"

template <typename _T> class Tensor {
public:
    Tensor(const std::vector<unsigned int>& shape, const _T value=0);
    Tensor(const _T value=0) : Tensor(std::vector<unsigned int>{}, value) {};
    Tensor(const std::vector<unsigned int>& shape, const std::vector<_T>& initData) :
        Tensor(shape) {
            data = std::make_shared<std::vector<_T>>(initData);
    }
    Tensor(const std::vector<unsigned int> inShape, const std::vector<unsigned int> inStrides,
    const std::shared_ptr<std::vector<_T>>& inData, const unsigned int inOffset);
    
    Tensor(const Tensor& other);
    Tensor<_T>& operator=(const Tensor<_T>& other);

    Tensor<_T> copy() const ;

    void show(unsigned int indent=0);

    unsigned int dim() const { return shape.size(); };

    Tensor<_T> T() const;

    Tensor<_T> collapse(unsigned int axis=0) const;

    template <typename Tf>
    friend Tensor<Tf> cat(const std::vector< Tensor<Tf> >& tensors, unsigned int axis);
    template <typename Tf>
    friend Tensor<Tf> stack(const std::vector< Tensor<Tf> >& tensors, unsigned int axis);

    _T operator()(const std::vector<unsigned int>& indices) const;
    _T operator()(unsigned int index) const;
    Tensor<_T> operator[](const std::vector<unsigned int>& indices) const;
    Tensor<_T> operator[](unsigned int index) const;
    Tensor slice(const std::vector<slice>& slices);

    Tensor<_T> operator+(const Tensor<_T>& other) const;
    Tensor<_T> operator-(const Tensor<_T>& other) const;
    Tensor<_T> operator*(const Tensor<_T>& other) const;
    Tensor<_T> operator/(const Tensor<_T>& other) const;

    std::shared_ptr<std::vector<_T>> data;

    unsigned int offset = 0; // Offset for views/slices

    std::vector <unsigned int> strides;
    std::vector <unsigned int> shape;
};

//MARK: - Implementation
// constructor
template <typename _T>
Tensor<_T>::Tensor(const std::vector<unsigned int>& shape, const _T value)
    : shape(shape) {
    int length = 1;
    if (!(shape.empty())) {
        strides.resize(shape.size());
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = length;
            length *= shape[i];
        }
    }
    data = std::make_shared<std::vector<_T>>(length, value);
}

// constructor
template <typename _T>
Tensor<_T>::Tensor(const std::vector<unsigned int> inShape, const std::vector<unsigned int> inStrides,
    const std::shared_ptr<std::vector<_T>>& inData, const unsigned int inOffset)
    : shape(inShape), strides(inStrides), data(inData), offset(inOffset) {}

// copy constructor
template <typename _T>
Tensor<_T>::Tensor(const Tensor<_T>& other) :
shape(other.shape), strides(other.strides), offset(other.offset), data(std::make_shared<std::vector<_T>> (*other.data)) {}

// assignment operator
template <typename _T>
Tensor<_T>& Tensor<_T>::operator=(const Tensor<_T>& other) {
    if (shape != other.shape) {
        shape = other.shape;
        strides = other.strides;
        offset = other.offset;
        data = std::make_shared<std::vector<_T>> (*other.data);
        return (*this);
    }
    if (dim() == 0){
        data->at(offset) = other.data->at(other.offset);
        return (*this);
    }
    for (uint32_t i=0; i < shape[0]; i++){
        (*this)[i] = other[i];
    }
    return (*this);
}

// return a deep copy of the tensor
template <typename _T>
Tensor<_T> Tensor<_T>::copy() const {
    Tensor<_T> result (shape);
    if (dim() == 0){
        result.data->at(result.offset) = data->at(offset);
        return result;
    }
    for (uint32_t i=0; i < shape[0]; i++){
        result[i] = (*this)[i].copy();
    }
    return result;
}

template <typename _T>
void Tensor<_T>::show(unsigned int indent) {
    if (shape.empty()) {
        std::cout << (*data)[offset] << std::endl;
        return;
    }
    if (dim() == 1) {
        std::cout << "[ ";
        for (uint32_t i = 0; i < shape[0]; i++) {
            std::cout << (*data)[offset + i * strides[0]] << " ";
        }
        std::cout << "]";
        if (!indent) std::cout << std::endl;
    } else {
        std::cout << "[";
        for (uint32_t i=0; i < shape[0]; i++) {
            (*this)[i].show(indent + 1);
            
            if (i < shape[0] - 1){
                std::cout << std::endl;
                for (unsigned int i = 0; i < indent+1; i++)
                    std::cout << " ";
            }
        }
        std::cout << "]";
        if (!indent) std::cout << std::endl;
    }
}

template <typename _T>
Tensor<_T> Tensor<_T>::T() const {
    if (dim() == 0){
        return *this;
    }
    auto newShape = shape;
    std::reverse(newShape.begin(), newShape.end());
    
    Tensor<_T> result (newShape);

    auto indices = std::vector<unsigned int>(dim(), 0);
    
    for (unsigned int step=0; step < shape[0]*strides[0]; step++) {
        for (int i=dim()-1; i >= 0; i--){
            indices[i] = ( step / strides[i] ) % ( shape[i] );
        }
        auto reversed = indices;
        std::reverse(reversed.begin(), reversed.end());
        
        result[ reversed ] = (*this)[indices];
    }
    return result;
}

//MARK: - Access
template <typename _T>
_T Tensor<_T>::operator()(const std::vector<unsigned int>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::out_of_range("Number of indices does not match tensor dimensions");
    }
    unsigned int index = 0;
    for (unsigned int i = 0; i < indices.size(); ++i) {
        index += indices[i] * strides[i];
    }
    return (*data)[index];
}

template <typename _T>
_T Tensor<_T>::operator()(unsigned int index) const {
    if (shape.size() != 1) {
        throw std::out_of_range("Tensor is not 1-dimensional : cannot use single index");
    }
    return (*data)[index * strides[0] + offset];
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator[](const std::vector<unsigned int>& indices) const{
    if (indices.size() > shape.size()) {
        throw std::out_of_range("Number of indices exceeds tensor dimensions");
    }
    unsigned int newOffset = offset;
    for (unsigned int i = 0; i < indices.size(); ++i) {
        newOffset += indices[i] * strides[i];
    }
    
    Tensor<_T> subtensor (
        std::vector<unsigned int>(shape.begin() + indices.size(), shape.end()),
        std::vector<unsigned int>(strides.begin() + indices.size(), strides.end()),
        data,
        newOffset
    );

    return subtensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator[](unsigned int index) const{
    if (shape.empty()) {
        throw std::out_of_range("Cannot index into a scalar tensor");
    }
    unsigned int newOffset = offset + index * strides[0];
    Tensor<_T> subtensor (
        std::vector<unsigned int>(shape.begin() + 1, shape.end()),
        std::vector<unsigned int>(strides.begin() + 1, strides.end()),
        data,
        newOffset
    );

    return subtensor;
}

//MARK: - Arithmetic
template <typename _T>
Tensor<_T> Tensor<_T>::operator+(const Tensor<_T>& other) const{
    Tensor<_T> result {*this};
    if (dim() < other.dim()){
        for (uint32_t i=0; i < other.shape[0]; i++){
            Tensor<_T> subtensor {other[i]};
            result[i] = *this + subtensor;
        }
        return result;
    }
    if (dim() > other.dim()){
        for (uint32_t i=0; i < shape[0]; i++){
            Tensor<_T> subtensor {(*this)[i]};
            result[i] = subtensor + other;
        }
        return result;
    }
    if (shape != other.shape){
        throw std::invalid_argument("Tensors must have the same shape for elementwise addition");
    }
    if (dim() == 0){
        result.data->at(result.offset) += other.data->at(other.offset);
        return result;
    }
    for (uint32_t i=0; i<shape[0]; i++){
        result[i] = (*this)[i] + other[i];
    }
    return result;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator-(const Tensor<_T>& other) const{
    Tensor<_T> result {*this};
    if (dim() < other.dim()){
        for (uint32_t i=0; i < other.shape[0]; i++){
            Tensor<_T> subtensor {other[i]};
            result[i] = *this - subtensor;
        }
        return result;
    } else if (dim() > other.dim()){
        for (uint32_t i=0; i < shape[0]; i++){
            Tensor<_T> subtensor {(*this)[i]};
            result[i] = subtensor - other;
        }
        return result;
    } else {
        if (shape != other.shape){
            throw std::invalid_argument("Tensors must have the same shape for elementwise substraction");
        }
        if (dim() == 0){
            result.data->at(result.offset) -= other.data->at(other.offset);
            return result;
        }
        for (uint32_t i=0; i<shape[0]; i++){
            result[i] = (*this)[i] - other[i];
        }
        return result;
    }
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator*(const Tensor<_T>& other) const{
    if (dim() < other.dim()){
        Tensor<_T> result (other);
        for (uint32_t i=0; i < other.shape[0]; i++){
            Tensor<_T> subtensor {other[i]};
            result[i] = *this * subtensor;
        }
        return result;
    }
    Tensor<_T> result {*this};
    if (dim() > other.dim()){
        for (uint32_t i=0; i < shape[0]; i++){
            Tensor<_T> subtensor {(*this)[i]};
            result[i] = subtensor * other;
        }
        return result;
    } else {
        if (shape != other.shape){
            throw std::invalid_argument("Tensors must have the same shape for elementwise multiplication");
        }
        if (dim() == 0){
            result.data->at(result.offset) *= other.data->at(other.offset);
            return result;
        }
        for (uint32_t i=0; i<shape[0]; i++){
            result[i] = (*this)[i] * other[i];
        }
        return result;
    }
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator/(const Tensor<_T>& other) const{
    Tensor<_T> result {*this};
    if (dim() < other.dim()){
        for (uint32_t i=0; i < other.shape[0]; i++){
            Tensor<_T> subtensor {other[i]};
            result[i] = *this / subtensor;
        }
        return result;
    } else if (dim() > other.dim()){
        for (uint32_t i=0; i < shape[0]; i++){
            Tensor<_T> subtensor {(*this)[i]};
            result[i] = subtensor / other;
        }
        return result;
    } else {
        if (shape != other.shape){
            throw std::invalid_argument("Tensors must have the same shape for elementwise division");
        }
        if (dim() == 0){
            result.data->at(result.offset) /= other.data->at(other.offset);
            return result;
        }
        for (uint32_t i=0; i<shape[0]; i++){
            result[i] = (*this)[i] / other[i];
        }
        return result;
    }
}

//MARK: - Dim manip
template <typename _T>
Tensor<_T> Tensor<_T>::collapse(unsigned int axis) const{
    if (axis > dim()){
        throw std::out_of_range("Axis out of range for collapse");
    }
    auto newShape = shape;
    newShape.erase(newShape.begin() + axis);
    auto newStrides = strides;
    newStrides.erase(newStrides.begin() + axis);
    Tensor<_T> result {newShape, newStrides, std::make_shared<std::vector<_T>>(*data), offset};

    if (axis == 0){
        for (uint32_t i=1; i < shape[0]; i++){
            result = result + (*this)[i];
        }
    } else {
        for (uint32_t i=0; i < shape[0]; i++){
            result[i] = (*this)[i].collapse(axis-1);
        }
    }
    return result;
}

//MARK: Friend
template <typename _T>
Tensor<_T> cat(const std::vector< Tensor<_T> >& tensors, unsigned int axis=0){
    if (axis >= tensors[0].dim()){
        throw std::out_of_range("Axis out of range for cat");
    }
    for (unsigned int i=1; i < tensors.size(); i++){

        if (tensors[i].dim() != tensors[0].dim()){
            throw std::invalid_argument("dimension mismatch");
        }

        for (unsigned int j=0; j < tensors[i].dim(); j++){
            if (j != axis && tensors[i].shape[j] != tensors[0].shape[j]){
                throw std::invalid_argument("Shapes are not compatible for concatenation");
            }
        }
    }
    auto newShape = tensors[0].shape;
    for (unsigned int i=1; i < tensors.size(); i++){
        newShape[axis] += tensors[i].shape[axis];
    }
    Tensor<_T> result {newShape, 0};

    if (axis==0){
        uint32_t offset = 0;
        for (unsigned int n=0; n < tensors.size(); n++){
            for (uint32_t i=0; i < tensors[n].shape[0]; i++){
                result[i + offset] = tensors[n][i];
            }
            offset += tensors[n].shape[0];
        }

    } else {
        for (uint32_t i=0; i < tensors[0].shape[0]; i++){
            std::vector< Tensor<_T> > subtensors;
            for (unsigned int n=0; n < tensors.size(); n++){
                subtensors.push_back(tensors[n][i]);
            }
            result[i] = cat(subtensors, axis-1);
        }
    }
    return result;
}

template <typename _T>
Tensor<_T> stack(const std::vector< Tensor<_T> >& tensors, unsigned int axis=0) {
    if (axis > tensors[0].dim()){
        throw std::out_of_range("Axis out of range for stack");
    }
    for (unsigned int i=1; i < tensors.size(); i++){

        if (tensors[i].dim() != tensors[0].dim()){
            throw std::invalid_argument("dimension mismatch");
        }

        for (unsigned int j=0; j < tensors[i].dim(); j++){
            if (tensors[i].shape[j] != tensors[0].shape[j]){
                throw std::invalid_argument("Shapes are not compatible for stacking");
            }
        }
    }
    auto newShape = tensors[0].shape;
    newShape.insert(newShape.begin() + axis, tensors.size());
    Tensor<_T> result {newShape, 0};

    if (axis==0){
        for (unsigned int n=0; n < tensors.size(); n++){
            result[n] = tensors[n];
        }
    } else {
        for (uint32_t i=0; i < tensors[0].shape[0]; i++){
            std::vector< Tensor<_T> > subtensors;
            for (unsigned int n=0; n < tensors.size(); n++){
                subtensors.push_back(tensors[n][i]);
            }
            result[i] = stack(subtensors, axis-1);
        }
    }
    return result;
}