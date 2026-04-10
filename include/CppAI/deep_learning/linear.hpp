#pragma once
#include <random>
#include "Arrays/cpp_arrays.hpp"
#include "module.hpp"

class Linear: public Module{
    void initialize_parameters(float weightMean, float weightStdDev, float biasMean, float biasStdDev) {
        // Initialize weights
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> weight_dist(weightMean, weightStdDev);

        for (unsigned int i = 0; i < outputSize; ++i) {
            for (unsigned int j = 0; j < inputSize; ++j) {
                weights[{i, j}] = weight_dist(gen);
            }
        }

        // Initialize biases
        if (biasStdDev == 0.0f) {
            for (unsigned int i = 0; i < outputSize; ++i) {
                biases[i] = biasMean;
            }
            return;
        }
        std::normal_distribution<float> bias_dist(biasMean, biasStdDev);

        for (unsigned int i = 0; i < outputSize; ++i) {
            biases[i] = bias_dist(gen);
        }
    }
public:
    unsigned int inputSize;
    unsigned int outputSize;
    Array<float> prevInput;
    Array<float> biases;
    Array<float> weights;

    ~Linear() = default;

    Linear(unsigned int inInputSize, unsigned int inOutputSize, std::string initialization = "he") :
        inputSize(inInputSize), outputSize(inOutputSize) {

        biases = Array<float>({outputSize}, 0.0);
        weights = Array<float>({outputSize, inputSize});
        prevInput = Array<float>(inputSize);

        if (initialization == "lecun") {
            float weightStdDev = std::sqrt(1.0f / inputSize);
            initialize_parameters(0.0f, weightStdDev, 0.0f, 0.0f);
        } else if (initialization == "glorot" || initialization == "xavier") {
            float weightStdDev = std::sqrt(2.0f / (inputSize + outputSize));
            initialize_parameters(0.0f, weightStdDev, 0.0f, 0.0f);
        } else if (initialization == "he") {
            float weightStdDev = std::sqrt(2.0f / inputSize);
            initialize_parameters(0.0f, weightStdDev, 0.0f, 0.0f);
        } else {
            throw std::invalid_argument("Unsupported initialization method: " + initialization);
        }
    }
	std::vector< Array<float>* > get_parameters() override {
        return {&weights, &biases};
    }
    Array<float> forward(const Array<float>& input) override;
    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override;

    std::shared_ptr<Module> copy() const override {
        return std::make_shared<Linear>(*this);
    }
};