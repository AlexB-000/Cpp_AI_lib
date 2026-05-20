cmake_minimum_required(VERSION 3.28)
project(Arrays LANGUAGES CXX)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

add_library(arrays INTERFACE)
target_include_directories(arrays
INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/include
)
target_compile_features(arrays INTERFACE cxx_std_17)