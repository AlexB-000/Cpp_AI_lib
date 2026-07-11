#pragma once
#include <thread>
#include "Arrays/ndarrays.hpp"
#include "Dataset.hpp"
#include "./deep_learning/network.hpp"

namespace ClassMetrics {
    short bin_accuracy_sample(Network* net, const Array<float> *input, const Array<float> *target, bool around_zero=false){
        Array<float> output = net->forward(*input);
        float predicted;
        if (around_zero){
            predicted = output.at(0) >= 0.0f ? 1.0f : -1.0f;
        }
        else{
            predicted = output.at(0) >= 0.5f ? 1.0f : 0.0f;
        }
        return predicted == (*target).at(0) ? 1 : 0;
    }

    void bin_accuracy_thread(Network* net, const std::vector<Array<float>>* inputs, const std::vector<Array<float>>* targets,
            std::vector<short>* results, unsigned int start_index, unsigned int count, bool around_zero=false){
        for (unsigned int i=0; i<count; i++){
            (*results)[start_index + i] = bin_accuracy_sample(
                net, &(*inputs)[start_index + i], &(*targets)[start_index + i], around_zero
            );
        }
    }

    float bin_accuracy(Network& net, const std::vector<Array<float>>& X, const std::vector<Array<float>>& y,
        bool multithreading=true, bool around_zero=false){
        
        unsigned int correct = 0;
        std::vector<short> thread_results(X.size());
        std::vector<std::thread> threads;

        uint32_t thread_count = std::thread::hardware_concurrency() - 1;
        if (!multithreading) thread_count = 1;
        unsigned int samples_per_thread = (X.size() + thread_count -1) / thread_count;

        for (uint32_t i=0; i<X.size(); i+= samples_per_thread){
            unsigned int batch_size = std::min(samples_per_thread, static_cast<uint32_t>(X.size() - i));
            threads.emplace_back(bin_accuracy_thread, &net, &X, &y, &thread_results, i, batch_size, around_zero);
        }
        for (std::thread& t : threads){
            t.join();
        }
        for (short result : thread_results){
            correct += result;
        }
        return static_cast<float>(correct) / static_cast<float>(X.size());
    }

    void bin_accuracy_thread_dataset(Network* net, const Dataset* dataset,
            std::vector<short>* results, unsigned int start_index, unsigned int count, bool around_zero=false){
        for (unsigned int i=0; i<count; i++){
            (*results)[start_index + i] = bin_accuracy_sample(
                net, &(*dataset)[start_index + i][0], &(*dataset)[start_index + i][1], around_zero
            );
        }
    }

    float bin_accuracy(Network& net, const Dataset& dataset, bool multithreading=true, bool around_zero=false){
        unsigned int correct = 0;
        std::vector<short> thread_results(dataset.size());
        std::vector<std::thread> threads;

        uint32_t thread_count = std::thread::hardware_concurrency() - 1;
        if (!multithreading) thread_count = 1;
        unsigned int samples_per_thread = (dataset.size() + thread_count - 1) / thread_count;

        for (uint32_t i=0; i<dataset.size(); i+= samples_per_thread){
            unsigned int batch_size = std::min(samples_per_thread, static_cast<uint32_t>(dataset.size() - i));
            threads.emplace_back(bin_accuracy_thread_dataset,
                &net, &dataset, &thread_results, i, batch_size, around_zero
            );
        }
        for (std::thread& t : threads){
            t.join();
        }
        for (short result : thread_results){
            correct += result;
        }
        return static_cast<float>(correct) / static_cast<float>(dataset.size());
    }


    //MARK: Multiclass accuracy functions
    short multiclass_accuracy_sample(Network* net, const Array<float> *input, const Array<float> *target){
        Array<float> output = net->forward(*input);
        unsigned int predicted = 0;
        for (unsigned int j=1; j < output.shape[0]; j++){
            if (output.at(j) > output.at(predicted)){
                predicted = j;
            }
        }
        for (unsigned int j=0; j < target->shape[0]; j++){
            if ((*target).at(j) == 1.0f){
                if (j == predicted)
                    return 1;
                else
                    return 0;
            }
        }
        return 0;
    }

    void multiclass_accuracy_thread(Network* net, const std::vector<Array<float>>* inputs, const std::vector<Array<float>>* targets,
            std::vector<short>* results, unsigned int start_index, unsigned int count){
        for (unsigned int i=0; i<count; i++){
            (*results)[start_index + i] = multiclass_accuracy_sample(net, &(*inputs)[start_index + i], &(*targets)[start_index + i]);
        }
    }

    float multiclass_accuracy(Network& net, const std::vector<Array<float>>& X, const std::vector<Array<float>>& y,
        bool multithreading=true){
        
        unsigned int correct = 0;
        std::vector<short> thread_results(X.size());
        std::vector<std::thread> threads;

        uint32_t thread_count = std::thread::hardware_concurrency() - 1;
        if (!multithreading) thread_count = 1;
        unsigned int samples_per_thread = (X.size() + thread_count -1) / thread_count;

        for (uint32_t i=0; i<X.size(); i+= samples_per_thread){
            unsigned int batch_size = std::min(samples_per_thread, static_cast<uint32_t>(X.size() - i));
            threads.emplace_back(multiclass_accuracy_thread, &net, &X, &y, &thread_results, i, batch_size);
        }
        for (std::thread& t : threads){
            t.join();
        }
        for (short result : thread_results){
            correct += result;
        }
        return static_cast<float>(correct) / static_cast<float>(X.size());
    }

    void multiclass_accuracy_thread_dataset(Network* net, const Dataset* dataset,
            std::vector<short>* results, unsigned int start_index, unsigned int count){
        for (unsigned int i=0; i<count; i++){
            (*results)[start_index + i] = multiclass_accuracy_sample(net, &(*dataset)[start_index + i][0], &(*dataset)[start_index + i][1]);
        }
    }

    float multiclass_accuracy(Network& net, const Dataset& dataset, bool multithreading=true){
        unsigned int correct = 0;
        std::vector<short> thread_results(dataset.size());
        std::vector<std::thread> threads;

        uint32_t thread_count = std::thread::hardware_concurrency() - 1;
        if (!multithreading) thread_count = 1;
        unsigned int samples_per_thread = (dataset.size() + thread_count - 1) / thread_count;

        for (uint32_t i=0; i<dataset.size(); i+= samples_per_thread){
            unsigned int batch_size = std::min(samples_per_thread, static_cast<uint32_t>(dataset.size() - i));
            threads.emplace_back(multiclass_accuracy_thread_dataset, &net, &dataset, &thread_results, i, batch_size);
        }
        for (std::thread& t : threads){
            t.join();
        }
        for (short result : thread_results){
            correct += result;
        }
        return static_cast<float>(correct) / static_cast<float>(dataset.size());
    }
};