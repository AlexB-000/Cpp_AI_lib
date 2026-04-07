#pragma once
#include <vector>
#include "Arrays/cpp_arrays.hpp"

class Dataset{
protected:
    size_t _size;
public:
    Dataset(size_t inSize): _size(inSize){}

    size_t size() const { return _size; }
    virtual std::vector<Array<float>> operator[](size_t index) const = 0;
};

class ArrayDataset: public Dataset{
    std::vector<Array<float>> data;
    std::vector<Array<float>> target;

public:
    ArrayDataset(std::vector<Array<float>> X, std::vector<Array<float>> y): Dataset(X.size()), data(X), target(y) {}

    virtual std::vector<Array<float>> operator[](size_t index) const override {
        return {data[index], target[index]};
    }
};