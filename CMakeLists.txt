cmake_minimum_required(VERSION 3.28)
project(Cpp_AI_lib LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(BUILD_EXAMPLES "Build example programs" OFF)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

add_subdirectory(Arrays)

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
target_link_libraries(CppAI PUBLIC ndarrays)
target_compile_features(CppAI PUBLIC cxx_std_17)
target_compile_options(CppAI PUBLIC -O3 -march=native)

include(CheckIPOSupported)

if(PROJECT_IS_TOP_LEVEL)
    check_ipo_supported(RESULT ipo_supported)

    if(ipo_supported)
        message(STATUS "IPO enabled")
        set_property(TARGET CppAI PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        if(BUILD_EXAMPLES)
                set_property(TARGET example_affine example_XOR example_dataset_XOR PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        endif()
    else()
        message(STATUS "IPO not supported")
    endif()
endif()