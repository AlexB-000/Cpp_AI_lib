#pragma once
#include <random>
#include "Arrays/ndarrays.hpp"
#include "module.hpp"

enum class Padding {
    Zero,
    Average,
    Replication,
    Circular,
    Reflection
};

class Conv2d: public Module{
    void initialize_filters(float weightMean, float weightStdDev, int32_t seed=-1) {
        // Initialize weights
        if (seed == -1) {
            std::random_device rd;
            seed = rd();
        }
        std::mt19937 gen(seed);
        std::normal_distribution<float> weight_dist(weightMean, weightStdDev);
        
        for (uint32_t f=0; f < features; ++f) {
            for (uint32_t i = 0; i < filterSize[0]; ++i) {
                for (uint32_t j = 0; j < filterSize[1]; ++j) {
                    uint32_t offset = i * filters[f].strides[0] + j * filters[f].strides[1];
                    std::vector<uint32_t> indices (filterSize.size()-2, 0);
                    bool computing = true;
                    while (computing) {
                        uint32_t flatIdx = offset;
                        for (uint32_t k=0; k < filterSize.size()-2; ++k) {
                            flatIdx += indices[k] * filters[f].strides[2+k];
                        }
                        (*filters[f].data_ptr)[flatIdx] = weight_dist(gen);
                        // Increment indices
                        for (uint32_t k=filterSize.size()-3; k >=0 ; --k) {
                            if (++indices[k] < inputShape[2+k]) {
                                break;
                            }
                            indices[k] = 0;
                            if (k == 0) {
                                computing = false;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
public:
    std::vector<unsigned int> inputShape;
    Padding padding;
    unsigned int features;
    std::vector<unsigned int> filterSize;
    std::vector<unsigned int> stride;

    std::vector<Array<float>> filters;
    std::vector<unsigned int> outputShape;

    std::vector<Array<float>*> parameters;

    Array<float> inputCache;

    ~Conv2d() = default;

    Conv2d(const Conv2d& other) :
        inputShape(other.inputShape),
        padding(other.padding),
        features(other.features),
        filterSize(other.filterSize),
        stride(other.stride),
        filters(other.filters),
        outputShape(other.outputShape),
        inputCache(other.inputCache)
    {
        parameters.resize(features);
        for (unsigned int i = 0; i < features; ++i) {
            parameters[i] = &filters[i];
        }
    }

    Conv2d(
        std::vector<unsigned int> inInputShape,
        Padding inPadding,
        unsigned int inFeatures,
        std::vector<unsigned int> inFilterSize,
        std::vector<unsigned int> inStride,
        std::string initialization = "he", int32_t seed=-1
    ) :
        inputShape(inInputShape),
        padding(inPadding),
        features(inFeatures),
        filterSize(inFilterSize),
        stride(inStride),
        filters(std::vector<Array<float>>(features, Array<float>(filterSize))),
        inputCache(inputShape)
    {
        parameters.resize(features);
        for (unsigned int i = 0; i < features; ++i) {
            parameters[i] = &filters[i];
        }
        outputShape = {
            (inputShape[0]-1) / stride[0] + 1,
            (inputShape[1]-1) / stride[1] + 1,
            features
        };
        uint32_t input_flat_size = 1;
        for (const auto& dim : inputShape) {
            input_flat_size *= dim;
        }
        uint32_t output_flat_size = 1;
        for (const auto& dim : outputShape) {
            output_flat_size *= dim;
        }
        if (initialization == "lecun") {
            float weightStdDev = std::sqrt(1.0f / input_flat_size);
            initialize_filters(0.0f, weightStdDev, seed);
        } else if (initialization == "glorot" || initialization == "xavier") {
            float weightStdDev = std::sqrt(2.0f / (input_flat_size + output_flat_size));
            initialize_filters(0.0f, weightStdDev, seed);
        } else if (initialization == "he") {
            float weightStdDev = std::sqrt(2.0f / input_flat_size);
            initialize_filters(0.0f, weightStdDev, seed);
        } else {
            throw std::invalid_argument("Unsupported initialization method: " + initialization);
        }
    }
	std::vector< Array<float>* > get_parameters() override {
        return parameters;
    }
    Array<float> forward(const Array<float>& input) override;
    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override;

    std::shared_ptr<Module> copy() const override {
        return std::make_shared<Conv2d>(*this);
    }
};