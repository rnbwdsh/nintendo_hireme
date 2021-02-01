cmake_minimum_required(VERSION 3.19)
project(nintendo_hireme_cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(nintendo_hireme_cuda main.cu main.h )

set_target_properties(
        nintendo_hireme_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -Xptxas -O3")