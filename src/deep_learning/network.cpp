#include "../../include/deep_learning/network.hpp"

void Network::stackLayer(const std::shared_ptr<Module> layer){
	modules.push_back(layer);
	std::vector< Array<float>* > layerParams = layer->get_parameters();
	parameters.insert(parameters.end(), layerParams.begin(), layerParams.end());
}

Array<float> Network::forward(const Array<float>& input){
	Array<float> output = input;
	for(auto& layer : modules){
		output = layer->forward(output);
	}
	return output;
}

std::vector< Array<float> > Network::backward(const Array<float>& prevDeriv){

	Array<float> deriv = prevDeriv;

	std::vector< Array<float> > allDerivs;
	for(int i = modules.size() - 1; i >= 0; i--){
		std::vector< Array<float> > layerDerivs = modules[i]->backward(deriv);
		deriv = layerDerivs.back();
		layerDerivs.pop_back();
		allDerivs.insert(allDerivs.begin(), layerDerivs.begin(), layerDerivs.end());
	}
	return allDerivs;
}

void Network::save(const std::string& filename) const {
	std::ofstream outFile(filename, std::ios::binary);
	if (!outFile) {
		std::cerr << "Error opening file for saving: " << filename << std::endl;
		return;
	}

	for (const Array<float>* param : parameters) {
		unsigned int size = param->data.size();
		for (unsigned int i = 0; i < size; i++) {
			float value = param->data[i];
			outFile.write(reinterpret_cast<const char*>(&value), sizeof(value));
		}
	}
}

void Network::load(const std::string& filename) {
	std::ifstream inFile(filename, std::ios::binary);
	if (!inFile) {
		std::cerr << "Error opening file for loading: " << filename << std::endl;
		return;
	}

	for (Array<float>* param : parameters) {
		unsigned int size = param->data.size();
		for (unsigned int i = 0; i < size; i++) {
			float value;
			inFile.read(reinterpret_cast<char*>(&value), sizeof(value));
			param->data[i] = value;
		}
	}
}