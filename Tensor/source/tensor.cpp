#include "../include/tensor.hpp"

void Tensor::_create_new_tensor(const std::vector<unsigned int>& inShape, const float value) {
    shape = inShape;
    uint32_t length = 1;
    if (!(shape.empty())) {
        strides.resize(shape.size());
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = length;
            length *= shape[i];
        }
    }
    data = std::make_shared<std::vector<float>>(length, value);
    _is_subtensor = false;
    offset = 0;
}

void Tensor::_deep_copy(const Tensor& other) {
    _create_new_tensor(other.shape, 0);
    (*this) = other;
}

//MARK: - Constructors
// full control constructor for internal use (views and subtensors)
Tensor::Tensor(const std::vector<unsigned int> inShape, const std::vector<unsigned int> inStrides,
    const std::shared_ptr<std::vector<float>>& inData, const unsigned int inOffset, bool is_subtensor):
    shape(inShape), strides(inStrides), data(inData), offset(inOffset), _is_subtensor(is_subtensor) {}


// constructor
Tensor::Tensor(const std::vector<unsigned int>& shape, const float value){
    _create_new_tensor(shape, value);
}

// copy constructor

Tensor::Tensor(const Tensor& other) :
shape(other.shape){
    _deep_copy(other);
}

//MARK: assignment
// assignment operator
Tensor& Tensor::operator=(const Tensor& other) {
    if (shape != other.shape) {
        _deep_copy(other);
        return (*this);
    }
    if (dim() == 1){
        for (uint32_t i=0; i < shape[0]; i++){
            (*this)[i] = other[i];
        }
        return (*this);
    }
    if (dim() == 0){
        data->at(offset) = other.data->at(other.offset);
        return (*this);
    }
    for (uint32_t i=0; i < shape[0]; i++){
        (*this).subtensor(i) = other.subtensor(i);
    }
    return (*this);
}

//MARK: copy
// return a deep copy of the tensor
Tensor Tensor::copy() const {
    Tensor result {shape};
    result._deep_copy(*this);
    return result;
}

//MARK: show
void Tensor::show(unsigned int indent) const {
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
            Tensor subtensor = this->subtensor(i);

            subtensor.show(indent + 1);
            
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

// MARK: transpose
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
        auto reversed_indices = indices;
        std::reverse(reversed_indices.begin(), reversed_indices.end());
        
        result[ reversed_indices ] = (*this)[indices];
    }
    return result;
}

//MARK: - Access

float& Tensor::operator[](const std::vector<unsigned int>& indices) const {
    if (shape.size() == 0) {
        return (*data)[offset];
    }
    if (indices.size() != shape.size()) {
        throw std::out_of_range("Number of indices does not match tensor dimensions");
    }
    unsigned int index = offset;
    for (unsigned int i = 0; i < indices.size(); ++i) {
        index += indices[i] * strides[i];
    }
    return (*data)[index];
}

Tensor Tensor::subtensor(unsigned int index) const {
    if (dim() == 0 || index >= shape[0]) {
        throw std::out_of_range("Index out of range for subtensor");
    }
    std::vector<unsigned int> newShape {shape.begin() + 1, shape.end()};
    std::vector<unsigned int> newStrides {strides.begin() + 1, strides.end()};
    unsigned int newOffset = offset + index * strides[0];
    return Tensor(newShape, newStrides, data, newOffset, true);
}

//MARK: - Arithmetic (+)

Tensor Tensor::operator+(const Tensor& other) const{
    if (dim() < other.dim()){
        Tensor result {other};
        for (uint32_t i=0; i < other.shape[0]; i++){
            Tensor sub_other = other.subtensor(i);
            result.subtensor(i) = *this + sub_other;
        }
        return result;
    }
    Tensor result {*this};
    if (dim() > other.dim()){
        for (uint32_t i=0; i < shape[0]; i++){
            Tensor sub_this = (*this).subtensor(i);
            result.subtensor(i) = sub_this + other;
        }
        return result;
    }
    if (shape != other.shape){
        throw std::invalid_argument("Tensors must have the same shape for elementwise addition");
    }
    if (!_is_subtensor && !other._is_subtensor){
        result = fast_elementwise_add(*this, other);
        return result;
    }
    if  (dim() == 1){
        for (uint32_t i=0; i<shape[0]; i++){
            result[i] = (*this)[i] + other[i];
        }
        return result;
    }
    if (dim() == 0){
        result.data->at(result.offset) = (*this).data->at(offset) + other.data->at(other.offset);
        return result;
    }
    for (uint32_t i=0; i<shape[0]; i++){
        result.subtensor(i) = (*this).subtensor(i) + other.subtensor(i);
    }

    return result;
}

//MARK: (-)
Tensor Tensor::operator-(const Tensor& other) const{
    if (dim() < other.dim()){
        Tensor result {other};
        for (uint32_t i=0; i < other.shape[0]; i++){
            Tensor sub_other = other.subtensor(i);
            result.subtensor(i) = *this - sub_other;
        }
        return result;
    }
    Tensor result {*this};
    if (dim() > other.dim()){
        for (uint32_t i=0; i < shape[0]; i++){
            Tensor sub_this = (*this).subtensor(i);
            result.subtensor(i) = sub_this - other;
        }
        return result;
    }
    if (shape != other.shape){
        throw std::invalid_argument("Tensors must have the same shape for elementwise substraction");
    }
    if (!_is_subtensor && !other._is_subtensor){
        result = fast_elementwise_sub(*this, other);
        return result;
    }
    if  (dim() == 1){
        for (uint32_t i=0; i<shape[0]; i++){
            result[i] = (*this)[i] - other[i];
        }
        return result;
    }
    if (dim() == 0){
        result.data->at(result.offset) = (*this).data->at(offset) - other.data->at(other.offset);
        return result;
    }
    for (uint32_t i=0; i<shape[0]; i++){
        result.subtensor(i) = (*this).subtensor(i) - other.subtensor(i);
    }

    return result;
}

//MARK: (*)
Tensor Tensor::operator*(const Tensor& other) const{
    if (dim() < other.dim()){
        Tensor result {other};
        for (uint32_t i=0; i < other.shape[0]; i++){
            Tensor sub_other = other.subtensor(i);
            result.subtensor(i) = *this * sub_other;
        }
        return result;
    }
    Tensor result {*this};
    if (dim() > other.dim()){
        for (uint32_t i=0; i < shape[0]; i++){
            Tensor sub_this = (*this).subtensor(i);
            result.subtensor(i) = sub_this * other;
        }
        return result;
    }
    if (shape != other.shape){
        throw std::invalid_argument("Tensors must have the same shape for elementwise multiplication");
    }
    if (!_is_subtensor && !other._is_subtensor){
        result = fast_elementwise_mul(*this, other);
        return result;
    }
    if  (dim() == 1){
        for (uint32_t i=0; i<shape[0]; i++){
            result[i] = (*this)[i] * other[i];
        }
        return result;
    }
    if (dim() == 0){
        result.data->at(result.offset) = (*this).data->at(offset) * other.data->at(other.offset);
        return result;
    }
    for (uint32_t i=0; i<shape[0]; i++){
        result.subtensor(i) = (*this).subtensor(i) * other.subtensor(i);
    }

    return result;
}

//MARK: (/)
Tensor Tensor::operator/(const Tensor& other) const{
    if (dim() < other.dim()){
        Tensor result {other};
        for (uint32_t i=0; i < other.shape[0]; i++){
            Tensor sub_other = other.subtensor(i);
            result.subtensor(i) = *this / sub_other;
        }
        return result;
    }
    Tensor result {*this};
    if (dim() > other.dim()){
        for (uint32_t i=0; i < shape[0]; i++){
            Tensor sub_this = (*this).subtensor(i);
            result.subtensor(i) = sub_this / other;
        }
        return result;
    }
    if (shape != other.shape){
        throw std::invalid_argument("Tensors must have the same shape for elementwise division");
    }
    if (!_is_subtensor && !other._is_subtensor){
        result = fast_elementwise_div(*this, other);
        return result;
    }
    if  (dim() == 1){
        for (uint32_t i=0; i<shape[0]; i++){
            result[i] = (*this)[i] / other[i];
        }
        return result;
    }
    if (dim() == 0){
        result.data->at(result.offset) = (*this).data->at(offset) / other.data->at(other.offset);
        return result;
    }
    for (uint32_t i=0; i<shape[0]; i++){
        result.subtensor(i) = (*this).subtensor(i) / other.subtensor(i);
    }

    return result;
}

//MARK: - Dim manip

Tensor Tensor::collapse(unsigned int axis) const{
    if (axis > dim()){
        throw std::out_of_range("Axis out of range for collapse");
    }
    auto newShape = shape;
    newShape.erase(newShape.begin() + axis);
    Tensor result {newShape, 0};

    if (axis == 0){
        for (uint32_t i=0; i < shape[0]; i++){
            result = result + (*this).subtensor(i);
        }
    } else {
        for (uint32_t i=0; i < shape[0]; i++){
            result.subtensor(i) = (*this).subtensor(i).collapse(axis-1);
        }
    }
    return result;
}

//MARK: Friend functions

Tensor cat(const std::vector< Tensor >& tensors, unsigned int axis){
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
                result.subtensor(i + offset) = tensors[n].subtensor(i);
            }
            offset += tensors[n].shape[0];
        }

    } else {
        for (uint32_t i=0; i < tensors[0].shape[0]; i++){
            std::vector< Tensor > subtensors;
            for (unsigned int n=0; n < tensors.size(); n++){
                subtensors.push_back(tensors[n].subtensor(i));
            }
            result.subtensor(i) = cat(subtensors, axis-1);
        }
    }
    return result;
}

Tensor stack(const std::vector< Tensor >& tensors, unsigned int axis) {
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
            result.subtensor(n) = tensors[n];
        }
    } else {
        for (uint32_t i=0; i < tensors[0].shape[0]; i++){
            std::vector< Tensor > subtensors;
            for (unsigned int n=0; n < tensors.size(); n++){
                subtensors.push_back(tensors[n].subtensor(i));
            }
            result.subtensor(i) = stack(subtensors, axis-1);
        }
    }
    return result;
}

//MARK: - Fast elementwise
Tensor fast_elementwise_add(const Tensor& A, const Tensor& B) {
    if (A.dim() != B.dim()){
        throw std::invalid_argument("Tensors must have the same number of dimensions for fast elementwise operation");
    }
    for (unsigned int i=0; i < A.dim(); i++){
        if (A.shape[i] != B.shape[i]){
            throw std::invalid_argument("Tensors must have the same shape for fast elementwise operation");
        }
    }
    Tensor result {A.shape, 0};
    for (unsigned int i=0; i < A.data->size(); i++){
        (*result.data)[i] = (*A.data)[i] + (*B.data)[i];
    }
    return result;
}

Tensor fast_elementwise_sub(const Tensor& A, const Tensor& B) {
    if (A.dim() != B.dim()){
        throw std::invalid_argument("Tensors must have the same number of dimensions for fast elementwise operation");
    }
    for (unsigned int i=0; i < A.dim(); i++){
        if (A.shape[i] != B.shape[i]){
            throw std::invalid_argument("Tensors must have the same shape for fast elementwise operation");
        }
    }
    Tensor result {A.shape, 0};
    for (unsigned int i=0; i < A.data->size(); i++){
        (*result.data)[i] = (*A.data)[i] - (*B.data)[i];
    }
    return result;
}

Tensor fast_elementwise_mul(const Tensor& A, const Tensor& B) {
    if (A.dim() != B.dim()){
        throw std::invalid_argument("Tensors must have the same number of dimensions for fast elementwise operation");
    }
    for (unsigned int i=0; i < A.dim(); i++){
        if (A.shape[i] != B.shape[i]){
            throw std::invalid_argument("Tensors must have the same shape for fast elementwise operation");
        }
    }
    Tensor result {A.shape, 0};
    for (unsigned int i=0; i < A.data->size(); i++){
        (*result.data)[i] = (*A.data)[i] * (*B.data)[i];
    }
    return result;
}

Tensor fast_elementwise_div(const Tensor& A, const Tensor& B) {
    if (A.dim() != B.dim()){
        throw std::invalid_argument("Tensors must have the same number of dimensions for fast elementwise operation");
    }
    for (unsigned int i=0; i < A.dim(); i++){
        if (A.shape[i] != B.shape[i]){
            throw std::invalid_argument("Tensors must have the same shape for fast elementwise operation");
        }
    }
    Tensor result {A.shape, 0};
    for (unsigned int i=0; i < A.data->size(); i++){
        if ((*B.data)[i] == 0){
            throw std::invalid_argument("Division by zero in fast elementwise division");
        }
        (*result.data)[i] = (*A.data)[i] / (*B.data)[i];
    }
    return result;
}