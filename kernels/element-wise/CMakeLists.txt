pybind11_add_module(element_wise MODULE kernels.cu)
target_link_libraries(element_wise PRIVATE CUDA::cudart torch-deps cuda_compile_flags) 