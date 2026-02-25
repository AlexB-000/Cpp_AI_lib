#pragma once
#include <fstream>
#include "../../Arrays/cpp_arrays.hpp"
#include "module.hpp"

class Network: public Module{
public:
    std::vector<std::shared_ptr<Module>> modules;
    unsigned int inputSize;
    unsigned int outputSize;
    //std::vector< Array<float> > layerInputs;
    std::vector< Array<float> * > parameters;

    ~Network() = default;
	
	Network(unsigned int inInputSize, unsigned int inOutputSize) : inputSize(inInputSize), outputSize(inOutputSize) {}
	
	void stackLayer(const std::shared_ptr<Module> layer);

	std::vector< Array<float>* > get_parameters() {return parameters;}
    Array<float> forward(const Array<float>& input) override;
    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override;

    void save(const std::string& filename) const;
    void load(const std::string& filename);
};