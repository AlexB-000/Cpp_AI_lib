#pragma once
#include "Arrays/ndarrays.hpp"
#include "module.hpp"

enum class Pooling {
    Max,
    Average
};

class Pool2dLayer: public Module{
public:
    Pooling poolingType;
    std::vector<unsigned int> inputShape;
    std::vector<unsigned int> poolingSize;
    std::vector<unsigned int> stride;
    std::vector<unsigned int> outputShape;

    Array<float> inputCache;
    Array<float> outputCache;

    ~Pool2dLayer() = default;

    Pool2dLayer(const Pool2dLayer& other) :
        poolingType(other.poolingType),
        inputShape(other.inputShape),
        poolingSize(other.poolingSize),
        stride(other.stride),
        outputShape(other.outputShape),
        inputCache(other.inputCache),
        outputCache(other.outputCache)
    {
    }

    Pool2dLayer(
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
        return std::make_shared<Pool2dLayer>(*this);
    }
};