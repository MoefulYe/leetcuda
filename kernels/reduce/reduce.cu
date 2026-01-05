//

#include <cassert>
#include <common.hpp>
#include <concepts>
#include <cuda_fp16.h>
#include <sys/stat.h>
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
__global__ auto reduce_scalar_kernel(const Scalar *__restrict__ input,
                                     Sum *__restrict__ output, int n) -> void {
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

template <typename Scalar, typename Sum = Scalar,
          const int NUM_ELEM_PER_BLOCK = 256>
__global__ auto reduce_pack_kernel(const Scalar *__restrict__ input,
                                   Sum *__restrict__ output, int n) -> void {
  constexpr auto PACK_SIZE = 128 / 8; // 按128位打包加载, 按字节计
  constexpr auto SCALAR_SIZE = sizeof(Scalar);
  static_assert(SCALAR_SIZE < PACK_SIZE,
                "Scalar size must be less than pack size");
  static_assert(PACK_SIZE % SCALAR_SIZE == 0,
                "Pack size must be multiple of scalar size");
  constexpr auto NUM_ELEMS_PER_PACK = PACK_SIZE / SCALAR_SIZE;

  constexpr auto NUM_THREADS_PER_BLOCK =
      NUM_ELEM_PER_BLOCK / NUM_ELEMS_PER_PACK;
  __shared__ Sum reduce_smem[(NUM_THREADS_PER_BLOCK + WARP_SIZE - 1) /
                             WARP_SIZE]; // 用于存储每个warp的部分和

  const auto tid = threadIdx.x;                // 线程在block内的id
  const auto bid = blockIdx.x;                 // block在grid内的id
  const auto warpid = threadIdx.x / WARP_SIZE; // warp在block内的id
  const auto laneid = threadIdx.x % WARP_SIZE; // 线程在warp内的id
  const auto idx = (blockDim.x * bid + tid) *
                   NUM_ELEMS_PER_PACK; // 线程的打包加载的第一个元素的id
  Scalar pack[NUM_ELEMS_PER_PACK];
  __128_BITS_MUT(pack[0]) = __128_BITS(input[idx]);
  auto thread_sum = ZERO<Sum>;
#pragma unroll
  for (auto i = 0; i < NUM_ELEMS_PER_PACK; i++) {
    const Sum val = (idx + i < n) ? static_cast<Sum>(pack[i])
                                  : ZERO<Sum>; // 读取输入数据，越界则为0
    thread_sum = thread_sum + val;
  }
  const Sum warp_sum = butterfly_reduce_sum(thread_sum); // 计算warp内的规约和
  if (laneid == 0) { // 每个warp的第一个线程将
    reduce_smem[warpid] = warp_sum;
  }
  __syncthreads(); // 同步所有线程，确保共享内存中的数据
  if (warpid == 0) {
    const Sum warp_sum =
        (laneid < (NUM_THREADS_PER_BLOCK + WARP_SIZE - 1) / WARP_SIZE)
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
__global__ auto reduce_vector_kernel(const Scalar *__restrict__ input,
                                     Sum *__restrict__ output, int n) -> void {
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