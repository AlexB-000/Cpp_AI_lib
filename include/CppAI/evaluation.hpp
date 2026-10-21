#pragma once
#include <thread>
#include "Arrays/ndarrays.hpp"
#include "Dataset.hpp"
#include "./deep_learning/network.hpp"

namespace ClassMetrics {
    short bin_accuracy_sample(
        Network* net,
        const Array<float> *input,
        const Array<float> *target,
        bool around_zero=false
    );

    void bin_accuracy_thread(
        Network* net,
        const std::vector<Array<float>>* inputs,
        const std::vector<Array<float>>* targets,
        std::vector<short>* results,
        unsigned int start_index, unsigned int count,
        bool around_zero=false
    );

    float bin_accuracy(
        Network& net,
        const std::vector<Array<float>>& X,
        const std::vector<Array<float>>& y,
        bool multithreading=true,
        bool around_zero=false
    );

    void bin_accuracy_thread_dataset(
        Network* net,
        const Dataset* dataset,
        std::vector<short>* results,
        unsigned int start_index,
        unsigned int count,
        bool around_zero=false
    );

    float bin_accuracy(
        Network& net,
        const Dataset& dataset,
        bool multithreading=true,
        bool around_zero=false
    );


    //MARK: Multiclass accuracy functions
    short multiclass_accuracy_sample(
        Network* net,
        const Array<float> *input,
        const Array<float> *target
    );

    void multiclass_accuracy_thread(
        Network* net,
        const std::vector<Array<float>>* inputs,
        const std::vector<Array<float>>* targets,
        std::vector<short>* results,
        unsigned int start_index, unsigned int count);

    float multiclass_accuracy(
        Network& net,
        const std::vector<Array<float>>& X,
        const std::vector<Array<float>>& y,
        bool multithreading=true
    );

    void multiclass_accuracy_thread_dataset(
        Network* net, const Dataset* dataset,
        std::vector<short>* results,
        unsigned int start_index, unsigned int count
    );

    float multiclass_accuracy(Network& net, const Dataset& dataset, bool multithreading=true);
};