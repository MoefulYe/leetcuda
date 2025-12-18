cmake_minimum_required(VERSION 3.10)
project(LeetCUDA LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(TORCH_CUDA_ARCH_LIST "8.6")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED PATHS ./.venv/lib/python3.12/site-packages/torch NO_DEFAULT_PATH)

add_library(torch_headers INTERFACE)
target_include_directories(torch_headers INTERFACE
    ${CMAKE_SOURCE_DIR}/.venv/lib/python3.12/site-packages/torch/include
    ${CMAKE_SOURCE_DIR}/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include
)

add_subdirectory(others)
add_subdirectory(kernels)