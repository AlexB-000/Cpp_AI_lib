#pragma once
#include <vector>
#include "Arrays/ndarrays.hpp"
#include "Dataset.hpp"

struct DataLoader{
protected:
    Dataset* dataset;
    unsigned int batch_size;
    unsigned int nb_batches;
public:
    DataLoader(Dataset* inDataset, size_t inBatchSize) : dataset(inDataset), batch_size(inBatchSize) {
        nb_batches = (dataset->size() + batch_size - 1) / batch_size;
    }

    size_t get_dataset_size() const { return dataset->size(); }

    size_t get_batch_size() const { return batch_size; }

    unsigned int get_batch_quantity() const { return nb_batches; }

    std::vector<std::vector<Array<float>>> get_batch(unsigned int batch_num){
        std::vector<std::vector<Array<float>>> batch;
        unsigned int start_index = batch_num * batch_size;
        for (unsigned int i = 0; i < batch_size; i++){
            if (start_index + i >= dataset->size()) {
                break; // Stop if we exceed the dataset size
            }
            batch.emplace_back((*dataset)[start_index + i]);
        }
        return batch;
    }
};