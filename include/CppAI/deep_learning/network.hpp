#pragma once
#include <fstream>
#include "Arrays/cpp_arrays.hpp"
#include "module.hpp"

class Network: public Module{
public:
    std::vector<std::shared_ptr<Module>> modules;
    // unsigned int inputSize;
    // unsigned int outputSize;
    //std::vector< Array<float> > layerInputs;
    std::vector< Array<float> * > parameters;

    ~Network() = default;
	
	Network() = default;

    Network(const Network& other) {
        modules.clear();
        parameters.clear();
        for (std::shared_ptr<Module> module : other.modules) {
            modules.push_back(module->copy());
            std::vector< Array<float>* > layerParams = modules.back()->get_parameters();
            parameters.insert(parameters.end(), layerParams.begin(), layerParams.end());
        }
    }

    Network& operator=(const Network& other){
        modules.clear();
        parameters.clear();
        for (std::shared_ptr<Module> module : other.modules) {
            modules.push_back(module->copy());
            std::vector< Array<float>* > layerParams = modules.back()->get_parameters();
            parameters.insert(parameters.end(), layerParams.begin(), layerParams.end());
        }
        return (*this);
    }

    std::shared_ptr<Module> copy() const override {
        return std::make_shared<Network>(*this);
    }
	
	void stackLayer(const std::shared_ptr<Module> layer);

	std::vector< Array<float>* > get_parameters() override {return parameters;}
    Array<float> forward(const Array<float>& input) override;
    std::vector< Array<float> > backward(const Array<float>& prevDeriv) override;

    void train_mode() {
        training = true;
        for (std::shared_ptr<Module> module : modules) {
            module->train_mode();
        }
    }
    void eval_mode() {
        training = false;
        for (std::shared_ptr<Module> module : modules) {
            module->eval_mode();
        }
    }

    void save(const std::string& filename) const;
    void load(const std::string& filename);
};