#include <iostream>
#include <random>
#include "CppAI/Cpp_AI.hpp"

#define SEED -1

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
        data.emplace_back(std::vector<uint32_t>{2}, std::vector<float>{value1, value2});
        if (value1 && value2 || (!value1 && !value2))
            target.emplace_back(std::vector<uint32_t>{2}, std::vector<float>{0.0f, 1.0f});
        else
            target.emplace_back(std::vector<uint32_t>{2}, std::vector<float>{1.0f, 0.0f});
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
            std::cout << output.at(0)-output.at(1) << " ";
        }
        std::cout << "\n";
    }
}

int main(){
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    auto t1 = high_resolution_clock::now();

    Network net;
    std::shared_ptr<Linear> layer1 = std::make_shared<Linear>(2, 5, "glorot");
    net.stackLayer(layer1);
    std::shared_ptr<PReLU> activation1 = std::make_shared<PReLU>(5);
    net.stackLayer(activation1);
    std::shared_ptr<Linear> layer2 = std::make_shared<Linear>(5, 2, "glorot");
    net.stackLayer(layer2);
    std::shared_ptr<Softmax> activation2 = std::make_shared<Softmax>(2);
    net.stackLayer(activation2);

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


    std::vector<std::vector<Array<float>>> data = generate_data(10000, SEED);
    std::vector<Array<float>> X = data[0];
    std::vector<Array<float>> y = data[1];

    std::vector<std::vector<Array<float>>> splited = train_test_split(X, y, 0.1f, SEED);
    std::vector<Array<float>> X_train = splited[0];
    std::vector<Array<float>> y_train = splited[1];
    std::vector<Array<float>> X_test = splited[2];
    std::vector<Array<float>> y_test = splited[3];

    std::cout << "## Data generated and splitted. Training size: " << X_train.size() << ", Test size: " << X_test.size() << "\n";

    CrossEntropyLoss loss(2);
    GD optimizer(&net, &loss);

    std::cout << "## Starting training...\n";

    optimizer.train(X_train, y_train, 10, 64, 0.1f, 2, false);

    params = net.get_parameters();
    std::cout << "Learned Parameters:\n";
    for (auto param : params) {
        param->show();
    }

    float train_accuracy = ClassMetrics::multiclass_accuracy(net, X_train, y_train);
    std::cout << "## Train Accuracy: " << train_accuracy * 100.0f << "%\n";

    float test_accuracy = ClassMetrics::multiclass_accuracy(net, X_test, y_test);
    std::cout << "## Test Accuracy: " << test_accuracy * 100.0f << "%\n";

    printDecisionGraph(net);

    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> exe_time = t2 - t1;

    std::cout << "Execution time: " << exe_time.count() << "ms\n";

    return 0;
}