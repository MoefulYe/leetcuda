pybind11_add_module(softmax MODULE softmax.cu)
target_link_libraries(softmax PRIVATE CUDA::cudart torch-deps cuda_compile_flags common) 