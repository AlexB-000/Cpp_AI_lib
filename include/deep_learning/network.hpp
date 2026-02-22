#pragma once
#include <fstream>
#include "../../Tensor/include/tensorMath.hpp"
#include "module.hpp"

class Network: public Module{
public:
    std::vector<std::shared_ptr<Module>> modules;
    unsigned int inputSize;
    unsigned int outputSize;
    //std::vector< Tensor > layerInputs;
    std::vector< Tensor * > parameters;

    ~Network() = default;
	
	Network(unsigned int inInputSize, unsigned int inOutputSize) : inputSize(inInputSize), outputSize(inOutputSize) {}
	
	void stackLayer(const std::shared_ptr<Module> layer);

	std::vector< Tensor* > get_parameters() {return parameters;}
    Tensor forward(const Tensor& input) override;
    std::vector< Tensor > backward(const Tensor& prevDeriv) override;

    void save(const std::string& filename) const;
    void load(const std::string& filename);
};