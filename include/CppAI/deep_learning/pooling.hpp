#pragma once
#include "Arrays/ndarrays.hpp"
#include "module.hpp"

enum class Pooling {
    Max,
    Average
};

class Pooling2d: public Module{
public:
    Pooling poolingType;
    std::vector<unsigned int> inputShape;
    std::vector<unsigned int> poolingSize;
    std::vector<unsigned int> stride;
    std::vector<unsigned int> outputShape;

    Array<float> inputCache;
    Array<float> outputCache;

    ~Pooling2d() = default;

    Pooling2d(const Pooling2d& other) :
        poolingType(other.poolingType),
        inputShape(other.inputShape),
        poolingSize(other.poolingSize),
        stride(other.stride),
        outputShape(other.outputShape),
        inputCache(other.inputCache),
        outputCache(other.outputCache)
    {
    }

    Pooling2d(
        Pooling inPoolingType,
        std::vector<unsigned int> inInputShape,
        std::vector<unsigned int> inPoolingSize,
        std::vector<unsigned int> inStride
    ) :
        poolingType(inPoolingType),
        inputShape(inInputShape),
        poolingSize(inPoolingSize),
        stride(inStride),
        inputCache(inInputShape),
        outputShape({
            (inputShape[0]-1) / stride[0] + 1,
            (inputShape[1]-1) / stride[1] + 1,
            inputShape[2]
        }),
        outputCache(outputShape)
    {
    }
	std::vector< Array<float>* > get_parameters() override {
        return {};
    }
    Array<float> forward(const Array<float>& input) override;
    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override;

    std::shared_ptr<Module> copy() const override {
        return std::make_shared<Pooling2d>(*this);
    }
};