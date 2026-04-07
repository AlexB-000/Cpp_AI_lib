cmake_minimum_required(VERSION 3.28)
project(Cpp_AI_lib LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(BUILD_EXAMPLES "Build example programs" OFF)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

add_library(arrays INTERFACE)
target_sources(arrays
INTERFACE
    FILE_SET HEADERS
    BASE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/Arrays
        ${CMAKE_CURRENT_LIST_DIR}/Arrays/include
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/Arrays/include/array_math.hpp
        ${CMAKE_CURRENT_LIST_DIR}/Arrays/include/array.hpp
        ${CMAKE_CURRENT_LIST_DIR}/Arrays/spp_arrays.hpp
)
target_compile_features(arrays INTERFACE cxx_std_17)

add_library(CppAI STATIC)
target_sources(CppAI
PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/src/deep_learning/linear.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/deep_learning/network.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/optimizers/GD.cpp
)
target_include_directories(CppAI
PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
)
target_link_libraries(CppAI PUBLIC arrays)
target_compile_features(CppAI PUBLIC cxx_std_17)
target_compile_options(CppAI PUBLIC -O3)