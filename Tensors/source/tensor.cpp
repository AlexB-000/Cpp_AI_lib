#include "../include/tensor.hpp"

// constructor
Tensor::Tensor(const std::vector<unsigned int>& shape, const float value)
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

Tensor::Tensor(const std::vector<unsigned int> inShape, const std::vector<unsigned int> inStrides,
    const std::shared_ptr<std::vector<_T>>& inData, const unsigned int inOffset)
    : shape(inShape), strides(inStrides), data(inData), offset(inOffset) {}

// copy constructor

Tensor::Tensor(const Tensor& other) :
shape(other.shape), strides(other.strides), offset(other.offset), data(std::make_shared<std::vector<_T>> (*other.data)) {}

// assignment operator

Tensor& Tensor::operator=(const Tensor& other) {
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

Tensor Tensor::copy() const {
    Tensor result (shape);
    if (dim() == 0){
        result.data->at(result.offset) = data->at(offset);
        return result;
    }
    for (uint32_t i=0; i < shape[0]; i++){
        result[i] = (*this)[i].copy();
    }
    return result;
}


void Tensor::show(unsigned int indent) {
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


Tensor Tensor::T() const {
    if (dim() == 0){
        return *this;
    }
    auto newShape = shape;
    std::reverse(newShape.begin(), newShape.end());
    
    Tensor result (newShape);

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

float Tensor::operator()(const std::vector<unsigned int>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::out_of_range("Number of indices does not match tensor dimensions");
    }
    unsigned int index = 0;
    for (unsigned int i = 0; i < indices.size(); ++i) {
        index += indices[i] * strides[i];
    }
    return (*data)[index];
}


float Tensor::operator()(unsigned int index) const {
    if (shape.size() != 1) {
        throw std::out_of_range("Tensor is not 1-dimensional : cannot use single index");
    }
    return (*data)[index * strides[0] + offset];
}


Tensor Tensor::operator[](const std::vector<unsigned int>& indices) const{
    if (indices.size() > shape.size()) {
        throw std::out_of_range("Number of indices exceeds tensor dimensions");
    }
    unsigned int newOffset = offset;
    for (unsigned int i = 0; i < indices.size(); ++i) {
        newOffset += indices[i] * strides[i];
    }
    
    Tensor subtensor (
        std::vector<unsigned int>(shape.begin() + indices.size(), shape.end()),
        std::vector<unsigned int>(strides.begin() + indices.size(), strides.end()),
        data,
        newOffset
    );

    return subtensor;
}


Tensor Tensor::operator[](unsigned int index) const{
    if (shape.empty()) {
        throw std::out_of_range("Cannot index into a scalar tensor");
    }
    unsigned int newOffset = offset + index * strides[0];
    Tensor subtensor (
        std::vector<unsigned int>(shape.begin() + 1, shape.end()),
        std::vector<unsigned int>(strides.begin() + 1, strides.end()),
        data,
        newOffset
    );

    return subtensor;
}

//MARK: - Arithmetic

Tensor Tensor::operator+(const Tensor& other) const{
    Tensor result {*this};
    if (dim() < other.dim()){
        for (uint32_t i=0; i < other.shape[0]; i++){
            Tensor subtensor {other[i]};
            result[i] = *this + subtensor;
        }
        return result;
    }
    if (dim() > other.dim()){
        for (uint32_t i=0; i < shape[0]; i++){
            Tensor subtensor {(*this)[i]};
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


Tensor Tensor::operator-(const Tensor& other) const{
    Tensor result {*this};
    if (dim() < other.dim()){
        for (uint32_t i=0; i < other.shape[0]; i++){
            Tensor subtensor {other[i]};
            result[i] = *this - subtensor;
        }
        return result;
    } else if (dim() > other.dim()){
        for (uint32_t i=0; i < shape[0]; i++){
            Tensor subtensor {(*this)[i]};
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


Tensor Tensor::operator*(const Tensor& other) const{
    if (dim() < other.dim()){
        Tensor result (other);
        for (uint32_t i=0; i < other.shape[0]; i++){
            Tensor subtensor {other[i]};
            result[i] = *this * subtensor;
        }
        return result;
    }
    Tensor result {*this};
    if (dim() > other.dim()){
        for (uint32_t i=0; i < shape[0]; i++){
            Tensor subtensor {(*this)[i]};
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


Tensor Tensor::operator/(const Tensor& other) const{
    Tensor result {*this};
    if (dim() < other.dim()){
        for (uint32_t i=0; i < other.shape[0]; i++){
            Tensor subtensor {other[i]};
            result[i] = *this / subtensor;
        }
        return result;
    } else if (dim() > other.dim()){
        for (uint32_t i=0; i < shape[0]; i++){
            Tensor subtensor {(*this)[i]};
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

Tensor Tensor::collapse(unsigned int axis) const{
    if (axis > dim()){
        throw std::out_of_range("Axis out of range for collapse");
    }
    auto newShape = shape;
    newShape.erase(newShape.begin() + axis);
    auto newStrides = strides;
    newStrides.erase(newStrides.begin() + axis);
    Tensor result {newShape, newStrides, std::make_shared<std::vector<_T>>(*data), offset};

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

Tensor cat(const std::vector< Tensor >& tensors, unsigned int axis=0){
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
    Tensor result {newShape, 0};

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
            std::vector< Tensor > subtensors;
            for (unsigned int n=0; n < tensors.size(); n++){
                subtensors.push_back(tensors[n][i]);
            }
            result[i] = cat(subtensors, axis-1);
        }
    }
    return result;
}


Tensor stack(const std::vector< Tensor >& tensors, unsigned int axis=0) {
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
    Tensor result {newShape, 0};

    if (axis==0){
        for (unsigned int n=0; n < tensors.size(); n++){
            result[n] = tensors[n];
        }
    } else {
        for (uint32_t i=0; i < tensors[0].shape[0]; i++){
            std::vector< Tensor > subtensors;
            for (unsigned int n=0; n < tensors.size(); n++){
                subtensors.push_back(tensors[n][i]);
            }
            result[i] = stack(subtensors, axis-1);
        }
    }
    return result;
}