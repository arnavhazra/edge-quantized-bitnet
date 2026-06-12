cmake_minimum_required(VERSION 3.18)
project(BitNetInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Build with CUDA support" OFF)

include_directories(include)

if(USE_CUDA)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    add_executable(bitnet_engine src/engine/main.cpp src/kernels/ternary_gemm.cu)
    set_target_properties(bitnet_engine PROPERTIES CUDA_SEPARATE_COMPILATION ON)
else()
    add_executable(bitnet_engine src/engine/main.cpp)
endif()
