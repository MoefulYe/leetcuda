#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pytypedefs.h>
#include <string>

float add_gpu(float a, float b);
std::string device_info();

namespace py = pybind11;

PYBIND11_MODULE(pybinding_demo, m) {
  m.doc() = "Minimal pybind11 + CUDA demo module (CMake AOT build)";

  m.def("device_info", &device_info, "Return CUDA device info string");
  m.def("add_gpu", &add_gpu, py::arg("a"), py::arg("b"),
        "Compute a+b on GPU and return result");
}
