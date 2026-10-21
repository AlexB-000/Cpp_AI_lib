#pragma once
#include <random>
#include <vector>
#include "Arrays/ndarrays.hpp"

std::vector<std::vector<Array<float>>> train_test_split(
    std::vector<Array<float>> X, std::vector<Array<float>> y,
    float test_prop=0.1f, int32_t seed=-1
);