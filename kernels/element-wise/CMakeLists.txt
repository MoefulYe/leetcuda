pybind11_add_module(element_wise MODULE kernels.cu)
target_link_libraries(element_wise PRIVATE CUDA::cudart torch-deps)
target_compile_options(
  element_wise
  PRIVATE
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    --expt-relaxed-constexpr
    --expt-extended-lambda
    --use_fast_math)
