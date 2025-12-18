pybind11_add_module(pybinding_demo MODULE cuda_ops.cu bindings.cpp)
target_link_libraries(pybinding_demo PRIVATE CUDA::cudart)