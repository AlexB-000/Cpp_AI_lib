cmake_minimum_required(VERSION 3.28)
project(examples LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(sum_product_example)
target_sources(sum_product_example PRIVATE sum_product_example.cpp)
target_link_libraries(sum_product_example PRIVATE CppAI)
target_compile_features(sum_product_example PRIVATE cxx_std_17)
target_compile_options(sum_product_example PRIVATE -O3)