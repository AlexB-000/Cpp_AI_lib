#pragma once
#include "Arrays/cpp_arrays.hpp"

class Module{
protected:
    bool training = false;
public:
    void train_mode() {
        training = true;
    }
    void eval_mode() {
        training = false;
    }
	virtual std::vector< Array<float>* > get_parameters() = 0;
    virtual Array<float> forward(const Array<float>& input) = 0;
    virtual std::vector< Array<float> > backward(const Array<float>& prevDeriv) = 0;

    virtual std::shared_ptr<Module> copy() const = 0;
};