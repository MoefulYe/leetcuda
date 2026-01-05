//

#include <cassert>
#include <common.hpp>
#include <concepts>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector_types.h>

constexpr auto WARP_SIZE = 32;

template <typename T>
__forceinline__ __device__ auto butterfly_reduce_sum(T val) -> T {
  for (auto mask = WARP_SIZE >> 1; mask > 0; mask = mask >> 1) {
    val = val + __shfl_xor_sync(__activemask(), val, mask);
  }
  return val;
}

template <typename Vector, typename Sum>
concept ThreadReduceable = requires(Vector v) {
  { thread_reduce<Vector, Sum>(v) } -> std::same_as<Sum>;
};

template <typename Vector, typename Sum>
auto __device__ __forceinline__ thread_reduce(Vector v) -> Sum = delete;

template <>
auto __device__ __forceinline__ thread_reduce<f32x4, f32>(f32x4 v) -> f32 {
  return v.x + v.y + v.z + v.w;
}

template <>
auto __device__ __forceinline__ thread_reduce<f16x2, f16>(f16x2 v) -> f16 {
  return v.x + v.y;
}

template <>
auto __device__ __forceinline__ thread_reduce<f16x2, f32>(f16x2 v) -> f32 {
  return f32{v.x} + f32{v.y};
}

template <>
auto __device__ __forceinline__ thread_reduce<b16x2, b16>(b16x2 v) -> b16 {
  return v.x + v.y;
}

template <>
auto __device__ __forceinline__ thread_reduce<b16x2, f32>(b16x2 v) -> f32 {
  return f32{v.x} + f32{v.y};
}

static_assert(ThreadReduceable<f32x4, f32>);
static_assert(ThreadReduceable<f16x2, f16>);
static_assert(ThreadReduceable<f16x2, f32>);
static_assert(ThreadReduceable<b16x2, b16>);
static_assert(ThreadReduceable<b16x2, f32>);

template <typename Scalar, typename Sum = Scalar,
          const int NUM_ELEM_PER_BLOCK = 256>
__global__ auto block_reduce_sum_kernel(const Scalar *__restrict__ input,
                                        Sum *__restrict__ output, int n)
    -> void {
  constexpr auto NUM_THREADS_PER_BLOCK = NUM_ELEM_PER_BLOCK;
  constexpr auto NUM_WARPS_PER_BLOCK = (NUM_THREADS_PER_BLOCK + WARP_SIZE - 1) /
                                       WARP_SIZE;  // 计算每个block中的warp数量
  __shared__ Sum reduce_smem[NUM_WARPS_PER_BLOCK]; // 用于存储每个warp的部分和
  // assert(NUM_THREADS_PER_BLOCK == blockDim.x);
  // assert(blockDim.x <= KWARP_SIZE * KWARP_SIZE);
  const auto tid = threadIdx.x;                // 线程在block内的id
  const auto bid = blockIdx.x;                 // block在grid内的id
  const auto warpid = threadIdx.x / WARP_SIZE; // warp在block内的id
  const auto laneid = threadIdx.x % WARP_SIZE; // 线程在warp内的id
  const auto idx = blockDim.x * bid + tid;     // 线程全局id

  const Sum val = (idx < n) ? input[idx] : ZERO<Sum>; // 读取输入数据，越界则为0
  const Sum sum = butterfly_reduce_sum(val);          // 计算warp内的规约和
  if (laneid == 0) { // 每个warp的第一个线程将部分和存入共享内存
    reduce_smem[warpid] = sum;
  }
  __syncthreads(); // 同步所有线程，确保共享内存中的数据可见
  if (warpid == 0) {
    const Sum warp_sum = (laneid < NUM_WARPS_PER_BLOCK)
                             ? reduce_smem[laneid]
                             : ZERO<Sum>; // 读取每个warp的部分和
    const Sum block_sum = butterfly_reduce_sum(warp_sum); // 计算block内的规约和
    if (laneid == 0) {
      atomicAdd(output, block_sum); // 使用原子操作将结果累加到输出
    }
  }
}

template <typename Scalar, typename Sum = Scalar, typename Vector,
          const int NUM_ELEM_PER_BLOCK = 256>
  requires IsVectorOf<Vector, Scalar> && ThreadReduceable<Vector, Sum>
__global__ auto block_reduce_sum_kernel(const Scalar *__restrict__ input,
                                        Sum *__restrict__ output, int n)
    -> void {
  static_assert(NUM_ELEM_PER_BLOCK % VectorTraits<Vector>::SIZE == 0,
                "NUM_ELEM_PER_BLOCK must be multiple of Vector::SIZE");
  constexpr auto NUM_THREADS_PER_BLOCK =
      NUM_ELEM_PER_BLOCK / VectorTraits<Vector>::SIZE;
  constexpr auto NUM_WARPS_PER_BLOCK =
      (NUM_THREADS_PER_BLOCK + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ Sum reduce_smem[NUM_WARPS_PER_BLOCK];

  // assert(NUM_THREADS_PER_BLOCK == blockDim.x);
  // assert(blockDim.x <= KWARP_SIZE * KWARP_SIZE);
  const auto tid = threadIdx.x;
  const auto bid = blockIdx.x;
  const auto idx = (blockDim.x * bid + tid) * VectorTraits<Vector>::SIZE;
  const Vector vec = (idx < n)
                         ? (reinterpret_cast<const Vector *>(&input[idx]))[0]
                         : ZERO<Vector>;
  const Sum thread_sum = thread_reduce<Vector, Sum>(vec);
  const auto warpid = threadIdx.x / WARP_SIZE;
  const auto laneid = threadIdx.x % WARP_SIZE;
  const Sum warp_sum = butterfly_reduce_sum(thread_sum);
  if (laneid == 0) {
    reduce_smem[warpid] = warp_sum;
  }
  __syncthreads();
  if (warpid == 0) {
    const Sum warp_sum =
        (laneid < NUM_WARPS_PER_BLOCK) ? reduce_smem[laneid] : ZERO<Sum>;
    const Sum block_sum = butterfly_reduce_sum(warp_sum);
    if (laneid == 0) {
      atomicAdd(output, block_sum);
    }
  }
}

template <const int NUM_THREADS_PER_BLOCK = 256 / 8>
__global__ auto block_all_reduce_sum_f16x8_pack_f16_kernel(
    const f16 *__restrict__ input, f16 *__restrict__ output, int n) -> void {
  constexpr auto NUM_WARPS_PER_BLOCK = (NUM_THREADS_PER_BLOCK + WARP_SIZE - 1) /
                                       WARP_SIZE;  // 计算每个block中的warp数量
  __shared__ f16 reduce_smem[NUM_WARPS_PER_BLOCK]; // 用于存储每个warp的部分和
  // assert(NUM_THREADS_PER_BLOCK == blockDim.x);
  // assert(blockDim.x <= KWARP_SIZE * KWARP_SIZE);
  const auto tid = threadIdx.x;                  // 线程在block内的id
  const auto bid = blockIdx.x;                   // block在grid内的id
  const auto warpid = threadIdx.x / WARP_SIZE;   // warp在block内的id
  const auto laneid = threadIdx.x % WARP_SIZE;   // 线程在warp内的id
  const auto idx = (blockDim.x * bid + tid) * 8; // 线程全局id
  f16 pack[8];
  F32X4_MUT(pack[0]) = F32X4(input[idx]);
  auto thread_sum = ZERO<f16>;
#pragma unroll
  for (auto i = 0; i < 8; i++) {
    const auto val =
        (idx + i < n) ? pack[i] : ZERO<f16>; // 读取输入数据，越界则为0
    thread_sum = thread_sum + val;
  }
  const auto sum = butterfly_reduce_sum(thread_sum); // 计算warp内的规约和
  if (laneid == 0) { // 每个warp的第一个线程将部分和存入共享内存
    reduce_smem[warpid] = sum;
  }
  __syncthreads(); // 同步所有线程，确保共享内存中的数据可见
  if (warpid == 0) {
    const auto warp_sum = (laneid < NUM_WARPS_PER_BLOCK)
                              ? f16{reduce_smem[laneid]}
                              : ZERO<f16>; // 读取每个warp的部分和
    const auto block_sum =
        butterfly_reduce_sum(warp_sum); // 计算block内的规约和
    if (laneid == 0) {
      atomicAdd(output, block_sum); // 使用原子操作将结果累加到输出
    }
  }
}

template <const int NUM_THREADS_PER_BLOCK = 256 / 8>
__global__ auto block_all_reduce_sum_f16x8_pack_f32_kernel(
    const f16 *__restrict__ input, f32 *__restrict__ output, int n) -> void {
  constexpr auto NUM_WARPS_PER_BLOCK = (NUM_THREADS_PER_BLOCK + WARP_SIZE - 1) /
                                       WARP_SIZE;  // 计算每个block中的warp数量
  __shared__ f32 reduce_smem[NUM_WARPS_PER_BLOCK]; // 用于存储每个warp的部分和
  // assert(NUM_THREADS_PER_BLOCK == blockDim.x);
  // assert(blockDim.x <= KWARP_SIZE * KWARP_SIZE);
  const auto tid = threadIdx.x;                  // 线程在block内的id
  const auto bid = blockIdx.x;                   // block在grid内的id
  const auto warpid = threadIdx.x / WARP_SIZE;   // warp在block内的id
  const auto laneid = threadIdx.x % WARP_SIZE;   // 线程在warp内的id
  const auto idx = (blockDim.x * bid + tid) * 8; // 线程全局id
  f16 pack[8];
  F32X4_MUT(pack[0]) = F32X4(input[idx]);
  auto thread_sum = ZERO<f32>;
#pragma unroll
  for (auto i = 0; i < 8; i++) {
    const auto val =
        (idx + i < n) ? f32{pack[i]} : ZERO<f32>; // 读取输入数据，越界则为0
    thread_sum = thread_sum + val;
  }
  const auto sum = butterfly_reduce_sum(thread_sum); // 计算warp内的规约和
  if (laneid == 0) { // 每个warp的第一个线程将部分和存入共享内存
    reduce_smem[warpid] = sum;
  }
  __syncthreads(); // 同步所有线程，确保共享内存中的数据可见
  if (warpid == 0) {
    const auto warp_sum = (laneid < NUM_WARPS_PER_BLOCK)
                              ? f32{reduce_smem[laneid]}
                              : ZERO<f32>; // 读取每个warp的部分和
    const auto block_sum =
        butterfly_reduce_sum(warp_sum); // 计算block内的规约和
    if (laneid == 0) {
      atomicAdd(output, block_sum); // 使用原子操作将结果累加到输出
    }
  }
}

// f8e4m3x16_pack-f16 sum
template <const int NUM_THREADS_PER_BLOCK = 256 / 16>
__global__ auto block_all_reduce_sum_f8e4m3x16_pack_f16_kernel(
    const f8e4m3 *__restrict__ input, f16 *__restrict__ output, int n) -> void {
  constexpr auto NUM_WARPS_PER_BLOCK =
      (NUM_THREADS_PER_BLOCK + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ f16 reduce_smem[NUM_WARPS_PER_BLOCK];
  const auto tid = threadIdx.x;
  const auto bid = blockIdx.x;
  const auto warpid = threadIdx.x / WARP_SIZE;
  const auto laneid = threadIdx.x % WARP_SIZE;
  const auto idx = (blockDim.x * bid + tid) * 16;
  f8e4m3 pack[16];
  __128_BITS_MUT(pack[0]) = __128_BITS(input[idx]);
  auto thread_sum = ZERO<f16>;
#pragma unroll
  for (auto i = 0; i < 16; i++) {
    const auto val = (idx + i < n) ? f16{pack[i]} : ZERO<f16>;
    thread_sum = thread_sum + val;
  }
  const auto sum = butterfly_reduce_sum(thread_sum);
  if (laneid == 0) {
    reduce_smem[warpid] = sum;
  }
  __syncthreads();
  if (warpid == 0) {
    const auto warp_sum =
        (laneid < NUM_WARPS_PER_BLOCK) ? f16{reduce_smem[laneid]} : ZERO<f16>;
    const auto block_sum = butterfly_reduce_sum(warp_sum);
    if (laneid == 0) {
      atomicAdd(output, block_sum);
    }
  }
}

// f8e5m2x16_pack-f16 sum
template <const int NUM_THREADS_PER_BLOCK = 256 / 16>
__global__ auto block_all_reduce_sum_f8e5m2x16_pack_f16_kernel(
    const f8e5m2 *__restrict__ input, f16 *__restrict__ output, int n) -> void {
  constexpr auto NUM_WARPS_PER_BLOCK =
      (NUM_THREADS_PER_BLOCK + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ f16 reduce_smem[NUM_WARPS_PER_BLOCK];
  const auto tid = threadIdx.x;
  const auto bid = blockIdx.x;
  const auto warpid = threadIdx.x / WARP_SIZE;
  const auto laneid = threadIdx.x % WARP_SIZE;
  const auto idx = (blockDim.x * bid + tid) * 16;
  f8e5m2 pack[16];
  __128_BITS_MUT(pack[0]) = __128_BITS(input[idx]);
  auto thread_sum = ZERO<f16>;
#pragma unroll
  for (auto i = 0; i < 16; i++) {
    const auto val = (idx + i < n) ? f16{pack[i]} : ZERO<f16>;
    thread_sum = thread_sum + val;
  }
  const auto sum = butterfly_reduce_sum(thread_sum);
  if (laneid == 0) {
    reduce_smem[warpid] = sum;
  }
  __syncthreads();
  if (warpid == 0) {
    const auto warp_sum =
        (laneid < NUM_WARPS_PER_BLOCK) ? f16{reduce_smem[laneid]} : ZERO<f16>;
    const auto block_sum = butterfly_reduce_sum(warp_sum);
    if (laneid == 0) {
      atomicAdd(output, block_sum);
    }
  }
}

// int
// i8x16_pack-i32 sum
template <const int NUM_THREADS_PER_BLOCK = 256 / 16>
__global__ auto block_all_reduce_sum_i8x16_pack_i32_kernel(
    const i8 *__restrict__ input, i32 *__restrict__ output, int n) -> void {
  constexpr auto NUM_WARPS_PER_BLOCK =
      (NUM_THREADS_PER_BLOCK + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ i32 reduce_smem[NUM_WARPS_PER_BLOCK];
  const auto tid = threadIdx.x;
  const auto bid = blockIdx.x;
  const auto warpid = threadIdx.x / WARP_SIZE;
  const auto laneid = threadIdx.x % WARP_SIZE;
  const auto idx = (blockDim.x * bid + tid) * 16;
  i8 pack[16];
  __128_BITS_MUT(pack[0]) = __128_BITS(input[idx]);
  auto thread_sum = ZERO<i32>;
#pragma unroll
  for (auto i = 0; i < 16; i++) {
    const auto val = (idx + i < n) ? static_cast<i32>(pack[i]) : ZERO<i32>;
    thread_sum = thread_sum + val;
  }
  const auto sum = butterfly_reduce_sum(thread_sum);
  if (laneid == 0) {
    reduce_smem[warpid] = sum;
  }
  __syncthreads();
  if (warpid == 0) {
    const auto warp_sum =
        (laneid < NUM_WARPS_PER_BLOCK) ? reduce_smem[laneid] : ZERO<i32>;
    const auto block_sum = butterfly_reduce_sum(warp_sum);
    if (laneid == 0) {
      atomicAdd(output, block_sum);
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

#define LANUCH_REDUCE_KERNEL(NT, packed_type, acc_type, element_type,          \
                             out_type)                                         \
  block_all_reduce_sum_##packed_type##_##acc_type##_kernel<(NT)>               \
      <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),        \
                        reinterpret_cast<out_type *>(y.data_ptr()), N);

#define DISPATCH_REDUCE_KERNEL(K, packed_type, acc_type, element_type,         \
                               n_elements, out_type)                           \
  const int NT = (K) / (n_elements);                                           \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (NT) {                                                                \
  case 32:                                                                     \
    LANUCH_REDUCE_KERNEL(32, packed_type, acc_type, element_type, out_type)    \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_REDUCE_KERNEL(64, packed_type, acc_type, element_type, out_type)    \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_REDUCE_KERNEL(128, packed_type, acc_type, element_type, out_type)   \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_REDUCE_KERNEL(256, packed_type, acc_type, element_type, out_type)   \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_REDUCE_KERNEL(512, packed_type, acc_type, element_type, out_type)   \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_REDUCE_KERNEL(1024, packed_type, acc_type, element_type, out_type)  \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error(                                                  \
        "only support (K)/(n_elements): 32/64/128/256/512/1024");              \
    break;                                                                     \
  }

#define LANUCH_REDUCE_VECTOR_KERNEL(NT, scalar_type, acc_type, vector_type,    \
                                    n_elements, out_type)                      \
  block_reduce_sum_kernel<scalar_type, acc_type, vector_type,                  \
                          (NT) * (n_elements)>                                 \
      <<<grid, block>>>(reinterpret_cast<scalar_type *>(x.data_ptr()),         \
                        reinterpret_cast<out_type *>(y.data_ptr()), N);

#define DISPATCH_REDUCE_VECTOR_KERNEL(K, scalar_type, acc_type, vector_type,   \
                                      n_elements, out_type)                    \
  const int NT = (K) / (n_elements);                                           \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (NT) {                                                                \
  case 32:                                                                     \
    LANUCH_REDUCE_VECTOR_KERNEL(32, scalar_type, acc_type, vector_type,        \
                                n_elements, out_type)                          \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_REDUCE_VECTOR_KERNEL(64, scalar_type, acc_type, vector_type,        \
                                n_elements, out_type)                          \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_REDUCE_VECTOR_KERNEL(128, scalar_type, acc_type, vector_type,       \
                                n_elements, out_type)                          \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_REDUCE_VECTOR_KERNEL(256, scalar_type, acc_type, vector_type,       \
                                n_elements, out_type)                          \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_REDUCE_VECTOR_KERNEL(512, scalar_type, acc_type, vector_type,       \
                                n_elements, out_type)                          \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_REDUCE_VECTOR_KERNEL(1024, scalar_type, acc_type, vector_type,      \
                                n_elements, out_type)                          \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error(                                                  \
        "only support (K)/(n_elements): 32/64/128/256/512/1024");              \
    break;                                                                     \
  }

#define TORCH_BINDING_REDUCE(packed_type, acc_type, th_type, element_type,     \
                             n_elements, out_type)                             \
  torch::Tensor block_all_reduce_sum_##packed_type##_##acc_type(               \
      torch::Tensor x) {                                                       \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    auto y_th_type =                                                           \
        (th_type) == torch::kInt8 ? torch::kInt32 : torch::kFloat32;           \
    auto options =                                                             \
        torch::TensorOptions().dtype(y_th_type).device(torch::kCUDA, 0);       \
    auto y = torch::zeros({1}, options);                                       \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(1024 / (n_elements));                                         \
      dim3 grid((N + 1024 - 1) / 1024);                                        \
      block_all_reduce_sum_##packed_type##_##acc_type##_kernel<1024 /          \
                                                               (n_elements)>   \
          <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),    \
                            reinterpret_cast<out_type *>(y.data_ptr()), N);    \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        DISPATCH_REDUCE_KERNEL(K, packed_type, acc_type, element_type,         \
                               n_elements, out_type)                           \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(1024 / (n_elements));                                       \
        dim3 grid((N + 1024 - 1) / 1024);                                      \
        block_all_reduce_sum_##packed_type##_##acc_type##_kernel<1024 /        \
                                                                 (n_elements)> \
            <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),  \
                              reinterpret_cast<out_type *>(y.data_ptr()), N);  \
      }                                                                        \
    }                                                                          \
    return y;                                                                  \
  }

#define TORCH_BINDING_REDUCE_VECTOR(packed_type, acc_name, scalar_type,        \
                                    vector_type, acc_type, th_type,            \
                                    n_elements, out_type)                      \
  torch::Tensor block_all_reduce_sum_##packed_type##_##acc_name(               \
      torch::Tensor x) {                                                       \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    auto y_th_type =                                                           \
        (th_type) == torch::kInt8 ? torch::kInt32 : torch::kFloat32;           \
    auto options =                                                             \
        torch::TensorOptions().dtype(y_th_type).device(torch::kCUDA, 0);       \
    auto y = torch::zeros({1}, options);                                       \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(1024 / (n_elements));                                         \
      dim3 grid((N + 1024 - 1) / 1024);                                        \
      block_reduce_sum_kernel<scalar_type, acc_type, vector_type, 1024>        \
          <<<grid, block>>>(reinterpret_cast<scalar_type *>(x.data_ptr()),     \
                            reinterpret_cast<out_type *>(y.data_ptr()), N);    \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        DISPATCH_REDUCE_VECTOR_KERNEL(K, scalar_type, acc_type, vector_type,   \
                                      n_elements, out_type)                    \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(1024 / (n_elements));                                       \
        dim3 grid((N + 1024 - 1) / 1024);                                      \
        block_reduce_sum_kernel<scalar_type, acc_type, vector_type, 1024>      \
            <<<grid, block>>>(reinterpret_cast<scalar_type *>(x.data_ptr()),   \
                              reinterpret_cast<out_type *>(y.data_ptr()), N);  \
      }                                                                        \
    }                                                                          \
    return y;                                                                  \
  }

// packed_type, acc_type, th_type, element_type, n_elements_per_pack, out_type
TORCH_BINDING_REDUCE(f32, f32, torch::kFloat32, f32, 1, f32)
TORCH_BINDING_REDUCE_VECTOR(f32x4, f32, f32, f32x4, f32, torch::kFloat32, 4,
                            f32)
TORCH_BINDING_REDUCE(f16, f16, torch::kHalf, f16, 1, f32)
TORCH_BINDING_REDUCE(f16, f32, torch::kHalf, f16, 1, f32)
TORCH_BINDING_REDUCE_VECTOR(f16x2, f16, f16, f16x2, f16, torch::kHalf, 2, f32)
TORCH_BINDING_REDUCE_VECTOR(f16x2, f32, f16, f16x2, f32, torch::kHalf, 2, f32)
TORCH_BINDING_REDUCE(f16x8_pack, f16, torch::kHalf, f16, 8, f32)
TORCH_BINDING_REDUCE(f16x8_pack, f32, torch::kHalf, f16, 8, f32)
TORCH_BINDING_REDUCE(bf16, bf16, torch::kBFloat16, b16, 1, f32)
TORCH_BINDING_REDUCE(bf16, f32, torch::kBFloat16, b16, 1, f32)
TORCH_BINDING_REDUCE_VECTOR(bf16x2, bf16, b16, b16x2, b16, torch::kBFloat16, 2,
                            f32)
TORCH_BINDING_REDUCE_VECTOR(bf16x2, f32, b16, b16x2, f32, torch::kBFloat16, 2,
                            f32)
TORCH_BINDING_REDUCE(bf16x8_pack, bf16, torch::kBFloat16, b16, 8, f32)
TORCH_BINDING_REDUCE(bf16x8_pack, f32, torch::kBFloat16, b16, 8, f32)
TORCH_BINDING_REDUCE(fp8_e4m3, f16, torch::kFloat8_e4m3fn, __nv_fp8_storage_t,
                     1, f32)
TORCH_BINDING_REDUCE(fp8_e4m3x16_pack, f16, torch::kFloat8_e4m3fn,
                     __nv_fp8_storage_t, 16, f32)
TORCH_BINDING_REDUCE(fp8_e5m2, f16, torch::kFloat8_e5m2, __nv_fp8_storage_t, 1,
                     f32)
TORCH_BINDING_REDUCE(fp8_e5m2x16_pack, f16, torch::kFloat8_e5m2,
                     __nv_fp8_storage_t, 16, f32)
TORCH_BINDING_REDUCE(i8, i32, torch::kInt8, int8_t, 1, int32_t)
TORCH_BINDING_REDUCE(i8x16_pack, i32, torch::kInt8, int8_t, 16, int32_t)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32x4_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x8_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x8_pack_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x8_pack_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x8_pack_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e4m3_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e4m3x16_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e5m2_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e5m2x16_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_i8_i32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_i8x16_pack_i32)
}
