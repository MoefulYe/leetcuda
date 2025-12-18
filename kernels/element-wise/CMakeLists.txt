pybind11_add_module(element_wise MODULE kernels.cu)
find_library(
  TORCH_PYTHON_LIBRARY
  NAMES torch_python
  PATHS "${TORCH_INSTALL_PREFIX}/lib" "${CMAKE_SOURCE_DIR}/.venv/lib/python3.12/site-packages/torch/lib"
  NO_DEFAULT_PATH
)

if(NOT TORCH_PYTHON_LIBRARY)
  message(FATAL_ERROR "torch_python library not found; set TORCH_INSTALL_PREFIX or adjust the search path.")
endif()

target_link_libraries(element_wise PRIVATE CUDA::cudart torch_headers ${TORCH_LIBRARIES}
                                      ${TORCH_PYTHON_LIBRARY})
