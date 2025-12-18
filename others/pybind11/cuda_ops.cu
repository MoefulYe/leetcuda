#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <string>

float add_gpu(float a, float b);
std::string device_info();

namespace {

inline void throw_on_cuda_error(cudaError_t status, const char *where) {
  if (status == cudaSuccess) {
    return;
  }
  std::stringstream ss;
  ss << where << ": " << cudaGetErrorString(status);
  throw std::runtime_error(ss.str());
}

__global__ void add_kernel(float *out, float a, float b) { out[0] = a + b; }

} // namespace

float add_gpu(float a, float b) {
  float *d_out = nullptr;
  throw_on_cuda_error(cudaMalloc(&d_out, sizeof(float)), "cudaMalloc");

  add_kernel<<<1, 1>>>(d_out, a, b);
  throw_on_cuda_error(cudaGetLastError(), "add_kernel launch");
  throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  float out = 0.0f;
  throw_on_cuda_error(
      cudaMemcpy(&out, d_out, sizeof(float), cudaMemcpyDeviceToHost),
      "cudaMemcpy D2H");
  throw_on_cuda_error(cudaFree(d_out), "cudaFree");
  return out;
}

std::string device_info() {
  int device = 0;
  throw_on_cuda_error(cudaGetDevice(&device), "cudaGetDevice");

  cudaDeviceProp prop{};
  throw_on_cuda_error(cudaGetDeviceProperties(&prop, device),
                      "cudaGetDeviceProperties");

  std::ostringstream oss;
  oss << "device=" << device << " name=" << prop.name << " cc=" << prop.major
      << "." << prop.minor << " smCount=" << prop.multiProcessorCount
      << " memGB="
      << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
  return oss.str();
}
