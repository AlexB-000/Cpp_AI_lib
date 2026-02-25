#pragma once
#include <random>
#include "../../Arrays/cpp_arrays.hpp"
#include "module.hpp"

class Linear: public Module{
public:
    unsigned int inputSize;
    unsigned int outputSize;
    Array<float> prevInput;
    Array<float> biases;
    Array<float> weights;

    ~Linear() = default;

    Linear(unsigned int inInputSize, unsigned int inOutputSize, float initWeightScale = 1.0f, float initBiasScale = 1.0f) :
        inputSize(inInputSize), outputSize(inOutputSize) {

        biases = Array<float>({outputSize}, 0.0);
        weights = Array<float>({outputSize, inputSize});
        prevInput = Array<float>(inputSize);

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
	std::vector< Array<float>* > get_parameters() override;
    Array<float> forward(const Array<float>& input) override;
    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override;
};