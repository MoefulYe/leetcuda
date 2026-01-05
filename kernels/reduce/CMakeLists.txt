pybind11_add_module(reduce MODULE reduce.cu)
target_link_libraries(reduce PRIVATE CUDA::cudart torch-deps cuda_compile_flags common) 