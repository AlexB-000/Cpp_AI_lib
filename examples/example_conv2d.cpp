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
    std::uniform_int_distribution<> dis(0, 9);

    Network target_net;
    std::shared_ptr<Conv2d> layer1 = std::make_shared<Conv2d>(
        std::vector<unsigned int>{20, 20, 1}, // input shape (height, width, channels)
        Padding::Replication, // padding type
        1, // number of filters
        std::vector<unsigned int>{3, 3, 1}, // filter size (height, width, channels)
        std::vector<unsigned int>{2, 2}, // stride (height, width)
        "he", SEED // initialization method and seed
    );
    target_net.stackLayer(layer1);
    std::shared_ptr<LeakyReLU> activation1 = std::make_shared<LeakyReLU>(std::vector<uint32_t>{10, 10, 1}, 0.01f);
    target_net.stackLayer(activation1);
    std::shared_ptr<Pooling2d> pooling1 = std::make_shared<Pooling2d>(
        Pooling::Max,
        std::vector<unsigned int>{10, 10, 1},
        std::vector<unsigned int>{2, 2},
        std::vector<unsigned int>{2, 2}
    );
    target_net.stackLayer(pooling1);
    std::shared_ptr<Flatten> flatten1 = std::make_shared<Flatten>(std::vector<uint32_t>{5, 5, 1});
    target_net.stackLayer(flatten1);

    *target_net.parameters[0] = Array<float>(std::vector<uint32_t>{3, 3, 1}, std::vector<float>{
        -1.0f, -1.0f, -1.0f,
        -1.0f, 8.0f, -1.0f,
        -1.0f, -1.0f, -1.0f
    });
    target_net.eval_mode();

    std::vector< Array<float>* > params = target_net.get_parameters();
    std::cout << "Target Parameters:\n";
    for (auto param : params) {
        param->show();
    }

    for (unsigned int i=0; i<size; i++){
        uint32_t x = dis(gen);
        uint32_t y = dis(gen);
        Array<float> input({20, 20, 1}, 0.0f);
        for (uint32_t j=0; j<x; j++){
            for (uint32_t k=0; k<y; k++){
                (*input.data_ptr)[j * input.strides[0] + k * input.strides[1]] = 1.0f;
            }
        }
        data.emplace_back(input);
        Array<float> output = target_net.forward(input);
        target.emplace_back(output);
    }
    return {data, target};
}

int main(){
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    auto t1 = high_resolution_clock::now();

    Network net;
    std::shared_ptr<Conv2d> layer1 = std::make_shared<Conv2d>(
        std::vector<unsigned int>{20, 20, 1}, // input shape (height, width, channels)
        Padding::Replication, // padding type
        1, // number of filters
        std::vector<unsigned int>{3, 3, 1}, // filter size (height, width)
        std::vector<unsigned int>{2, 2}, // stride (height, width)
        "he", SEED // initialization method and seed
    );
    net.stackLayer(layer1);
    std::shared_ptr<LeakyReLU> activation1 = std::make_shared<LeakyReLU>(std::vector<uint32_t>{10, 10, 1}, 0.01f);
    net.stackLayer(activation1);
    std::shared_ptr<Pooling2d> pooling1 = std::make_shared<Pooling2d>(
        Pooling::Max,
        std::vector<unsigned int>{10, 10, 1},
        std::vector<unsigned int>{2, 2},
        std::vector<unsigned int>{2, 2}
    );
    net.stackLayer(pooling1);
    std::shared_ptr<Flatten> flatten1 = std::make_shared<Flatten>(std::vector<uint32_t>{5, 5, 1});
    net.stackLayer(flatten1);

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


    std::vector<std::vector<Array<float>>> data = generate_data(100000, SEED);    
    std::vector<Array<float>> X = data[0];
    std::vector<Array<float>> y = data[1];

    std::vector<std::vector<Array<float>>> splited = train_test_split(X, y, 0.1f, SEED);
    std::vector<Array<float>> X_train = splited[0];
    std::vector<Array<float>> y_train = splited[1];
    std::vector<Array<float>> X_test = splited[2];
    std::vector<Array<float>> y_test = splited[3];

    std::cout << "## Data generated and splitted. Training size: " << X_train.size() << ", Test size: " << X_test.size() << "\n";

    MSELoss loss{5*5};
    GD optimizer(&net, &loss);

    std::cout << "## Starting training...\n";

    optimizer.train(X_train, y_train, 10, 64, 0.01f, 2, false);

    params = net.get_parameters();
    std::cout << "Learned Parameters:\n";
    for (auto param : params) {
        param->show();
    }

    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> exe_time = t2 - t1;

    std::cout << "Execution time: " << exe_time.count() << "ms\n";

    return 0;
}