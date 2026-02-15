#pragma once
#include <random>
#include "../../Tensor/include/tensorMath.hpp"
#include "module.hpp"

class Linear: public Module{
public:
    unsigned int inputSize;
    unsigned int outputSize;
    Tensor prevInput;
    Tensor biases;
    Tensor weights;

    ~Linear() = default;

    Linear(unsigned int inInputSize, unsigned int inOutputSize, float initWeightScale = 1.0f, float initBiasScale = 1.0f) :
        inputSize(inInputSize), outputSize(inOutputSize) {

        biases = Tensor({outputSize}, 0.0);
        weights = Tensor({outputSize, inputSize});
        prevInput = Tensor(inputSize);

        // Initialize weights with small random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> weight_dist(-initWeightScale, initWeightScale);

        for (unsigned int i = 0; i < outputSize; ++i) {
            for (unsigned int j = 0; j < inputSize; ++j) {
                weights[{i, j}] = weight_dist(gen);
            }
        }

        // Initialize biases
        std::normal_distribution<float> bias_dist(-initBiasScale, initBiasScale);

        for (unsigned int i = 0; i < outputSize; ++i) {
            biases[i] = bias_dist(gen);
        }
    }
	std::vector< Tensor* > get_parameters() override;
    Tensor forward(const Tensor& input) override;
    std::vector< Tensor > backward(const Tensor& prevDeriv) override;
};