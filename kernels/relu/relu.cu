#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector_types.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<const int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<const float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<const half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<const __nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<const float4 *>(&(value))[0])

#define INT4_MUT(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4_MUT(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2_MUT(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2_MUT(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS_MUT(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ auto relu_f32_kernel(const float *__restrict__ input,
                                float *__restrict__ outputs, int N) -> void {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr auto ZERO = 0.0f;
  if (idx < N) {
    auto val = input[idx];
    outputs[idx] = cuda::std::fmaxf(val, ZERO);
  }
}

__global__ auto relu_f32x4_kernel(const float *__restrict__ input,
                                  float *__restrict__ output, int N) -> void {
  auto idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  constexpr auto ZERO = 0.0f;
  if (idx < N) {
    const auto [x, y, z, w] = FLOAT4(input[idx]);
    const auto res = float4{
        .x = cuda::std::fmaxf(x, ZERO),
        .y = cuda::std::fmaxf(y, ZERO),
        .z = cuda::std::fmaxf(z, ZERO),
        .w = cuda::std::fmaxf(w, ZERO),
    };
    FLOAT4_MUT(output[idx]) = res;
  }
}

__global__ auto relu_f16_kernel(const half *__restrict__ input,
                                half *__restrict__ output, int N) -> void {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto ZERO = __float2half(0.0f);
  if (idx < N) {
    const auto val = input[idx];
    const auto res = __hmax(val, ZERO);
    output[idx] = res;
  }
}

__global__ auto relu_f16x2_kernel(const half *__restrict__ input,
                                  half *__restrict__ output, int N) -> void {
  auto idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  const auto ZERO = __float2half(0.0f);
  if (idx < N) {
    const auto [x, y] = HALF2(input[idx]);
    const auto res = half2{
        __hmax(x, ZERO),
        __hmax(y, ZERO),
    };
    HALF2_MUT(output[idx]) = res;
  }
}

__global__ auto relu_f16x8_kernel(const half *__restrict__ input,
                                  half *__restrict__ output, int N) -> void {
  const auto idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  const auto ZERO = __float2half(0.0f);
  const auto input_01 = HALF2(input[idx + 0]);
  const auto input_23 = HALF2(input[idx + 2]);
  const auto input_45 = HALF2(input[idx + 4]);
  const auto input_67 = HALF2(input[idx + 6]);
  const auto res_01 = half2{
      __hmax(input_01.x, ZERO),
      __hmax(input_01.y, ZERO),
  };
  const auto res_23 = half2{
      __hmax(input_23.x, ZERO),
      __hmax(input_23.y, ZERO),
  };
  const auto res_45 = half2{
      __hmax(input_45.x, ZERO),
      __hmax(input_45.y, ZERO),
  };
  const auto res_67 = half2{
      __hmax(input_67.x, ZERO),
      __hmax(input_67.y, ZERO),
  };
  if ((idx + 0) < N) {
    HALF2_MUT(output[idx + 0]) = res_01;
  }
  if ((idx + 2) < N) {
    HALF2_MUT(output[idx + 2]) = res_23;
  }
  if ((idx + 4) < N) {
    HALF2_MUT(output[idx + 4]) = res_45;
  }
  if ((idx + 6) < N) {
    HALF2_MUT(output[idx + 6]) = res_67;
  }
}

__global__ auto relu_f16x8_pack_kernel(const half *__restrict__ input,
                                       half *__restrict__ output, int N)
    -> void {
  const auto idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  const auto ZERO = __float2half(0.0f);
  half pack_in[8], pack_out[8];
  LDST128BITS_MUT(pack_in[0]) = LDST128BITS(input[idx]);
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    const auto in_half2 = HALF2(pack_in[i]);
    const auto res_half2 = half2{
        __hmax(in_half2.x, ZERO),
        __hmax(in_half2.y, ZERO),
    };
    HALF2_MUT(pack_out[i]) = res_half2;
  }
  if ((idx + 7) < N) {
    LDST128BITS_MUT(output[idx]) = LDST128BITS(pack_out[0]);
  } else {
    for (int i = 0; idx + i < N; i++) {
      output[idx + i] = pack_out[i];
    }
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define TORCH_BINDING_RELU(PackedType, TorchType, ElementType, PACKED_N)       \
  auto relu_##PackedType(torch::Tensor input, torch::Tensor output) {          \
    CHECK_TORCH_TENSOR_DTYPE(input, (TorchType))                               \
    CHECK_TORCH_TENSOR_DTYPE(output, (TorchType))                              \
    const int ndim = input.dim();                                              \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= input.size(i);                                                    \
      }                                                                        \
      dim3 block(256 / (PACKED_N));                                            \
      dim3 grid((N + 256 - 1) / 256);                                          \
      relu_##PackedType##_kernel<<<grid, block>>>(                             \
          reinterpret_cast<const ElementType *>(input.data_ptr()),             \
          reinterpret_cast<ElementType *>(output.data_ptr()), N);              \
    } else {                                                                   \
      const int S = input.size(0);                                             \
      const int K = input.size(1);                                             \
      const int N = S * K;                                                     \
      if ((K / (N)) <= 1024) {                                                 \
        dim3 block(K / (PACKED_N));                                            \
        dim3 grid(S);                                                          \
        relu_##PackedType##_kernel<<<grid, block>>>(                           \
            reinterpret_cast<const ElementType *>(input.data_ptr()),           \
            reinterpret_cast<ElementType *>(output.data_ptr()), N);            \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= input.size(i);                                                  \
        }                                                                      \
        dim3 block(256 / (PACKED_N));                                          \
        dim3 grid((N + 256 - 1) / 256);                                        \
        relu_##PackedType##_kernel<<<grid, block>>>(                           \
            reinterpret_cast<const ElementType *>(input.data_ptr()),           \
            reinterpret_cast<ElementType *>(output.data_ptr()), N);            \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_RELU(f32, torch::kFloat32, float, 1)
TORCH_BINDING_RELU(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_RELU(f16, torch::kHalf, half, 1)
TORCH_BINDING_RELU(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_RELU(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_RELU(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(relu, m) {
  TORCH_BINDING_COMMON_EXTENSION(relu_f32)
  TORCH_BINDING_COMMON_EXTENSION(relu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x8_pack)
}
