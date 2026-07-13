cmake_minimum_required(VERSION 3.28)
project(examples LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(example_affine)
target_sources(example_affine PRIVATE example_affine.cpp)
target_link_libraries(example_affine PRIVATE CppAI)
target_compile_features(example_affine PRIVATE cxx_std_17)

add_executable(example_XOR)
target_sources(example_XOR PRIVATE example_XOR.cpp)
target_link_libraries(example_XOR PRIVATE CppAI)
target_compile_features(example_XOR PRIVATE cxx_std_17)

add_executable(example_dataset_XOR)
target_sources(example_dataset_XOR PRIVATE example_dataset_XOR.cpp)
target_link_libraries(example_dataset_XOR PRIVATE CppAI)
target_compile_features(example_dataset_XOR PRIVATE cxx_std_17)

add_executable(example_softmax)
target_sources(example_softmax PRIVATE example_softmax.cpp)
target_link_libraries(example_softmax PRIVATE CppAI)
target_compile_features(example_softmax PRIVATE cxx_std_17)