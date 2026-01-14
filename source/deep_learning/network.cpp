#include "../../include/deep_learning/network.hpp"

void Network::stackLayer(const std::shared_ptr<Module> layer){
	modules.push_back(layer);
	std::vector< Tensor* > layerParams = layer->get_parameters();
	parameters.insert(parameters.end(), layerParams.begin(), layerParams.end());
}

Tensor Network::forward(const Tensor& input){
	Tensor output = input;
	for(auto& layer : modules){
		output = layer->forward(output);
	}
	return output;
}

std::vector< Tensor > Network::backward(const Tensor& prevDeriv){

	Tensor deriv = prevDeriv;

	std::vector< Tensor > allDerivs;
	for(int i = modules.size() - 1; i >= 0; i--){
		std::vector< Tensor > layerDerivs = modules[i]->backward(deriv);
		deriv = layerDerivs.back();
		layerDerivs.pop_back();
		allDerivs.insert(allDerivs.begin(), layerDerivs.begin(), layerDerivs.end());
	}
	return allDerivs;
}