#include <iostream>
#include <random>
#include "CppAI/Cpp_AI.hpp"

std::vector<std::vector<Array<float>>> generate_data(unsigned int size, int32_t seed=-1){
    std::vector<Array<float>> data;
    std::vector<Array<float>> target;

    //random numbers generator
    if (seed == -1) {
        std::random_device rd;
        seed = rd();
    }
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, 1);

    for (unsigned int i=0; i<size; i++){
        float value1 = dis(gen);
        float value2 = dis(gen);
        data.push_back(Array<float>{{2}, {value1, value2}});
        if (value1 && value2 || (!value1 && !value2))
            target.push_back(Array<float>{{1}, {-1.0f}});
        else
            target.push_back(Array<float>{{1}, {1.0f}});
    }
    return {data, target};
}

void printDecisionGraph(Network& net){
    std::cout << std::fixed;
    std::cout.precision(2);
    std::cout << "## Decision Graph:\n";
    for (float x1 = 0.0f; x1 <= 1.0f; x1 += 0.2f){
        for (float x2 = 0.0f; x2 <= 1.0f; x2 += 0.2f){
            Array<float> input{{2}, {x1, x2}};
            Array<float> output = net.forward(input);
            std::cout << output[0] << " ";
        }
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]){
    bool loading = false;
    if (argc == 2 && std::string(argv[1]) == "--load"){
        loading = true;
    }

    Network net;
    std::shared_ptr<Linear> layer1 = std::make_shared<Linear>(2, 5, "glorot");
    net.stackLayer(layer1);
    std::shared_ptr<Tanh> activation1 = std::make_shared<Tanh>(5);
    net.stackLayer(activation1);
    std::shared_ptr<Linear> layer2 = std::make_shared<Linear>(5, 5, "glorot");
    net.stackLayer(layer2);
    std::shared_ptr<Tanh> activation2 = std::make_shared<Tanh>(5);
    net.stackLayer(activation2);
    std::shared_ptr<Linear> layer3 = std::make_shared<Linear>(5, 1, "glorot");
    net.stackLayer(layer3);
    std::shared_ptr<Tanh> activation3 = std::make_shared<Tanh>(1);
    net.stackLayer(activation3);

    if (loading) net.load("xor_dataset_model.bin");

    std::cout << "## Network created.\n";

    std::vector< Array<float>* > params = net.get_parameters();
    std::cout << "Initial Parameters:\n";
    for (auto param : params) {
        param->show();
    }
    for (size_t i = 0; i < params.size(); ++i){
        std::cout << "Parameter " << i << " shape: ";
        for (auto dim : params[i]->shape)
            std::cout << dim << " ";
        std::cout << "\n";
    }

    printDecisionGraph(net);

    std::vector<std::vector<Array<float>>> data = generate_data(1000);
    std::vector<Array<float>> X = data[0];
    std::vector<Array<float>> y = data[1];

    std::vector<std::vector<Array<float>>> splited = train_test_split(X, y, 0.1f);
    std::vector<Array<float>> X_train = splited[0];
    std::vector<Array<float>> y_train = splited[1];
    std::vector<Array<float>> X_test = splited[2];
    std::vector<Array<float>> y_test = splited[3];

    ArrayDataset train_dataset (X_train, y_train);

    std::cout << "## Data generated and splitted. Training size: " << X_train.size() << ", Test size: " << X_test.size() << "\n";

    MSELoss loss {1};
    GD optimizer(&net, &loss);

    DataLoader train_dataloader (&train_dataset, 100);
    optimizer.train(train_dataloader, 50, 1.0f, 1, true);

    std::cout << "## Training completed.\n";

    params = net.get_parameters();
    std::cout << "Learned Parameters:\n";
    for (auto param : params) {
        param->show();
    }

    float train_accuracy = ClassMetrics::bin_accuracy(net, X_train, y_train, true, true);
    std::cout << "## Train Accuracy: " << train_accuracy * 100.0f << "%\n";

    float test_accuracy = ClassMetrics::bin_accuracy(net, X_test, y_test, true, true);
    std::cout << "## Test Accuracy: " << test_accuracy * 100.0f << "%\n";

    printDecisionGraph(net);

    net.save("xor_dataset_model.bin");

    return 0;
}