#include "network.hpp"

void Network::stackLayer(const Module& layer){
	std::shared_ptr<Module> layerPtr = std::make_shared<Module>(layer);
	modules.push_back(layerPtr);
	parameters.push_back(layerPtr->parameters());
}

Tensor<double> Network::forward(const Tensor<double>& input){
	Tensor<double> output = input;
	for(auto& layer : modules){
		output = layer->forward(output);
	}
	return output;
}

std::vector< Tensor<double> > Network::backward(const Tensor<double>& prevDeriv){

	Tensor<double> deriv = prevDeriv;

	std::vector< Tensor<double> > allDerivs;
	for(int i = modules.size() - 1; i >= 0; i--){
		std::vector< Tensor<double> > layerDerivs = modules[i]->backward(deriv);
		deriv = layerDerivs.back();
		layerDerivs.pop_back();
		allDerivs.insert(allDerivs.begin(), layerDerivs.begin(), layerDerivs.end());
	}
	return allDerivs;
}