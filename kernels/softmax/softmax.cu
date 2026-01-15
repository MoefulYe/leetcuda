//

// Clang (including clangd) in CUDA mode pulls in CUDA headers via
// `__clang_cuda_runtime_wrapper.h`, which defines `__noinline__` as a macro.
// That macro breaks libstdc++ which uses `__attribute__((__noinline__, ...))`.
// Undefine it for clang-only parsing before including any libstdc++ headers.
#include <cmath>
#if defined(__clang__)
#ifdef __noinline__
#undef __noinline__
#endif
#endif

#include <cassert>
#include <sys/stat.h>
#include <torch/extension.h>
#include <torch/types.h>

#if defined(__clang__)
#ifdef __noinline__
#undef __noinline__
#endif
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <types.hpp>

constexpr auto WARP_SIZE = 32;
using namespace leetcuda;

template <typename T>
__forceinline__ __device__ auto butterfly_reduce(T val, auto &&bop) -> T {
  for (auto mask = WARP_SIZE >> 1; mask > 0; mask = mask >> 1) {
    val = bop(val, __shfl_xor_sync(__activemask(), val, mask));
  }
  return val;
}

template <typename T, const int NUM_THREADS>
__forceinline__ __device__ auto block_reduce(T val, auto &&bop, T identity)
    -> T {
  constexpr auto NUM_WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;
  __shared__ T shared_mem[NUM_WARPS_PER_BLOCK];
  const auto warp_id = threadIdx.x / WARP_SIZE;
  const auto lane_id = threadIdx.x % WARP_SIZE;
  const auto warp_reduced = butterfly_reduce(val, bop);
  if (lane_id == 0) {
    shared_mem[warp_id] = warp_reduced;
  }
  __syncthreads();
  const auto warp_reduced_ =
      (lane_id < NUM_WARPS_PER_BLOCK) ? shared_mem[lane_id] : identity;
  const auto block_reduced = butterfly_reduce(warp_reduced_, bop);
  return block_reduced;
}

enum class SoftmaxMode { Unsafe, Safe, Online };

template <typename Scalar, typename Acc = Scalar,
          const int NUM_ELEM_PER_BLOCK = 256,
          const SoftmaxMode MODE = SoftmaxMode::Safe>
__global__ auto softmax_scalar_kernel(const Scalar __restrict__ *x,
                                      Scalar __restrict__ *y, int N) -> void {
  constexpr auto NUM_THREADS_PER_BLOCK =
      NUM_ELEM_PER_BLOCK; // 每一个线程对应一个元素
  constexpr auto NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / WARP_SIZE;
  const auto block_id = blockIdx.x;
  const auto thread_id = threadIdx.x;
  const auto idx = block_id * NUM_ELEM_PER_BLOCK + thread_id;
  if constexpr (MODE == SoftmaxMode::Unsafe) {
    const Acc x_val =
        (idx < N) ? static_cast<Acc>(x[idx]) : lowest<Acc>(); // 防止越界访问
    const Acc exp_val =
        (idx < N) ? std::exp(static_cast<Acc>(x_val)) : zero<Acc>(); // 计算指数
    const Acc sum_exp_val = block_reduce<Acc, NUM_THREADS_PER_BLOCK>(
        exp_val, [](auto a, auto b) { return a + b; },
        zero<Acc>()); // 计算块内指数和
    const Acc y_val = exp_val / sum_exp_val;
    if (idx < N) {
      y[idx] = static_cast<Scalar>(y_val);
    }
  } else if constexpr (MODE == SoftmaxMode::Safe) {
    const Acc x_val = (idx < N) ? static_cast<Acc>(x[idx]) : lowest<Acc>();
    const Acc max_val = block_reduce<Acc, NUM_THREADS_PER_BLOCK>(
        x_val, [](auto a, auto b) { return std::max(a, b); }, lowest<Acc>());
    const Acc exp_val =
        (idx < N) ? std::exp(static_cast<Acc>(x_val) - max_val) : zero<Acc>();
    const Acc sum_exp_val = block_reduce<Acc, NUM_THREADS_PER_BLOCK>(
        exp_val, [](auto a, auto b) { return a + b; }, zero<Acc>());
    const Acc y_val = exp_val / sum_exp_val;
    if (idx < N) {
      y[idx] = static_cast<Scalar>(y_val);
    }
  } else {
    struct MD {
      Acc m; // 部分最大值
      Acc d; // 部分归一因子
    };
    const Acc x_val = (idx < N) ? static_cast<Acc>(x[idx]) : lowest<Acc>();
    const MD md = MD{.m = x_val, .d = (idx < N) ? one<Acc>() : zero<Acc>()};
    const auto [max, sum] = block_reduce<MD, NUM_ELEM_PER_BLOCK>(
        md,
        [](MD a, MD b) {
          const MD bigger = (a.m > b.m) ? a : b;
          const MD smaller = (a.m > b.m) ? b : a;
          return MD{.m = bigger.m,
                    .d = bigger.d + smaller.d * std::exp(smaller.m - bigger.m)};
        },
        MD{.m = lowest<Acc>(), .d = zero<Acc>()});
    const Acc y_val = std::exp(static_cast<Acc>(x_val) - max) / sum;
    if (idx < N) {
      y[idx] = static_cast<Scalar>(y_val);
    }
  }
}