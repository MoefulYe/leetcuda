pybind11_add_module(relu MODULE relu.cu)
target_link_libraries(relu PRIVATE CUDA::cudart torch-deps cuda_compile_flags common) 