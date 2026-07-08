cmake_minimum_required(VERSION 3.28)
project(Arrays LANGUAGES CXX)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

add_library(ndarrays INTERFACE)
target_sources(
    ndarrays
INTERFACE
    FILE_SET HEADERS
    BASE_DIRS
      ${CMAKE_CURRENT_LIST_DIR}/include
    FILES
      ${CMAKE_CURRENT_LIST_DIR}/include/Arrays/array.hpp
      ${CMAKE_CURRENT_LIST_DIR}/include/Arrays/array_math.hpp
      ${CMAKE_CURRENT_LIST_DIR}/include/Arrays/ndarrays.hpp
      ${CMAKE_CURRENT_LIST_DIR}/include/Arrays/utils.hpp
)
target_compile_features(ndarrays INTERFACE cxx_std_17)