"""
It says use C++17
Go build the matrix module
Go build the app

"""
cmake_minimum_required(VERSION 3.20)

project(CUDA_Neural_Network_Engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src/matrix)
add_subdirectory(apps)