# It says use C++17
# Go build the matrix module
# Go build the app

cmake_minimum_required(VERSION 3.20)

project(CUDA_Neural_Network_Engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_CUDA "Enables CUDA support" OFF)

#Enable CUDA when explictly requested:
#cmake -B build-cuda -DENABLE_CUDA=ON
if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17) 
    set(CMAKE_CUDA_STANDARD_REQUIRED ON) 
endif()

enable_testing()

if(ENABLE_CUDA)
    add_subdirectory(src/cuda)
endif()

add_subdirectory(src/matrix)
add_subdirectory(src/layers)
add_subdirectory(src/activations)
add_subdirectory(src/loss)
add_subdirectory(apps)
add_subdirectory(benchmarks)
add_subdirectory(tests)