#pragma once
#include <vector>
#include "../Tensor/include/tensor.hpp"

class Dataset{
protected:
    size_t _size;
public:
    Dataset(size_t inSize): _size(inSize){}

    size_t size() const { return _size; }
    virtual std::vector<Tensor> operator[](size_t index) const = 0;
};

class TensorDataset: public Dataset{
    std::vector<Tensor> data;
    std::vector<Tensor> target;

public:
    TensorDataset(std::vector<Tensor> X, std::vector<Tensor> y): Dataset(X.size()), data(X), target(y) {}

    virtual std::vector<Tensor> operator[](size_t index) const override {
        return {data[index], target[index]};
    }
};