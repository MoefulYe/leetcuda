//

// Clang (including clangd) in CUDA mode pulls in CUDA headers via
// `__clang_cuda_runtime_wrapper.h`, which defines `__noinline__` as a macro.
// That macro breaks libstdc++ which uses `__attribute__((__noinline__, ...))`.
// Undefine it for clang-only parsing before including any libstdc++ headers.
#if defined(__clang__)
#ifdef __noinline__
#undef __noinline__
#endif
#endif

#include <cassert>
#include <concepts>
#include <sys/stat.h>
#include <torch/extension.h>
#include <torch/types.h>

#if defined(__clang__)
#ifdef __noinline__
#undef __noinline__
#endif
#endif

#include <common.hpp>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr auto WARP_SIZE = 32;

template <typename T>
__forceinline__ __device__ auto butterfly_reduce(T val) -> T {
  for (auto mask = WARP_SIZE >> 1; mask > 0; mask = mask >> 1) {
    val = val + __shfl_xor_sync(__activemask(), val, mask);
  }
  return val;
}

template <typename Vector, typename Acc>
concept ThreadReduceable = requires(Vector v) {
  { thread_reduce<Vector, Acc>(v) } -> std::same_as<Acc>;
};

template <typename Vector, typename Acc>
auto __device__ __forceinline__ thread_reduce(Vector v) -> Acc = delete;

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
auto __device__ __forceinline__ thread_reduce<bf16x2, bf16>(bf16x2 v) -> bf16 {
  return v.x + v.y;
}

template <>
auto __device__ __forceinline__ thread_reduce<bf16x2, f32>(bf16x2 v) -> f32 {
  return f32{v.x} + f32{v.y};
}

static_assert(ThreadReduceable<f32x4, f32>);
static_assert(ThreadReduceable<f16x2, f16>);
static_assert(ThreadReduceable<f16x2, f32>);
static_assert(ThreadReduceable<bf16x2, bf16>);
static_assert(ThreadReduceable<bf16x2, f32>);

template <typename Scalar, typename Acc = Scalar,
          const int NUM_ELEM_PER_BLOCK = 256>
__global__ auto reduce_scalar_kernel(const Scalar *__restrict__ input,
                                     Acc *__restrict__ output, int n) -> void {
  constexpr auto NUM_THREADS_PER_BLOCK = NUM_ELEM_PER_BLOCK;
  constexpr auto NUM_WARPS_PER_BLOCK = (NUM_THREADS_PER_BLOCK + WARP_SIZE - 1) /
                                       WARP_SIZE;  // 计算每个block中的warp数量
  __shared__ Acc reduce_smem[NUM_WARPS_PER_BLOCK]; // 用于存储每个warp的部分和
  // assert(NUM_THREADS_PER_BLOCK == blockDim.x);
  // assert(blockDim.x <= KWARP_SIZE * KWARP_SIZE);
  const auto tid = threadIdx.x;                // 线程在block内的id
  const auto bid = blockIdx.x;                 // block在grid内的id
  const auto warpid = threadIdx.x / WARP_SIZE; // warp在block内的id
  const auto laneid = threadIdx.x % WARP_SIZE; // 线程在warp内的id
  const auto idx = blockDim.x * bid + tid;     // 线程全局id

  const Acc val = (idx < n) ? static_cast<Acc>(input[idx])
                            : ZERO<Acc>;      // 读取输入数据，越界则为0
  const Acc warp_acc = butterfly_reduce(val); // 计算warp内的规约和
  if (laneid == 0) { // 每个warp的第一个线程将部分和存入共享内存
    reduce_smem[warpid] = warp_acc;
  }
  __syncthreads(); // 同步所有线程，确保共享内存中的数据可见
  if (warpid == 0) {
    const Acc warp_acc = (laneid < NUM_WARPS_PER_BLOCK)
                             ? reduce_smem[laneid]
                             : ZERO<Acc>;             // 读取每个warp的部分和
    const Acc block_acc = butterfly_reduce(warp_acc); // 计算block内的规约和
    if (laneid == 0) {
      atomicAdd(output, block_acc); // 使用原子操作将结果累加到输出
    }
  }
}

template <typename Scalar, typename Acc = Scalar,
          const int NUM_ELEM_PER_BLOCK = 256>
__global__ auto reduce_pack_kernel(const Scalar *__restrict__ input,
                                   Acc *__restrict__ output, int n) -> void {
  constexpr auto PACK_SIZE = 128 / 8; // 按128位打包加载, 按字节计
  constexpr auto SCALAR_SIZE = sizeof(Scalar);
  static_assert(SCALAR_SIZE < PACK_SIZE,
                "Scalar size must be less than pack size");
  static_assert(PACK_SIZE % SCALAR_SIZE == 0,
                "Pack size must be multiple of scalar size");
  constexpr auto NUM_ELEMS_PER_PACK = PACK_SIZE / SCALAR_SIZE;

  constexpr auto NUM_THREADS_PER_BLOCK =
      NUM_ELEM_PER_BLOCK / NUM_ELEMS_PER_PACK;
  __shared__ Acc reduce_smem[(NUM_THREADS_PER_BLOCK + WARP_SIZE - 1) /
                             WARP_SIZE]; // 用于存储每个warp的部分和

  const auto tid = threadIdx.x;                // 线程在block内的id
  const auto bid = blockIdx.x;                 // block在grid内的id
  const auto warpid = threadIdx.x / WARP_SIZE; // warp在block内的id
  const auto laneid = threadIdx.x % WARP_SIZE; // 线程在warp内的id
  const auto idx = (blockDim.x * bid + tid) *
                   NUM_ELEMS_PER_PACK; // 线程的打包加载的第一个元素的id
  Scalar pack[NUM_ELEMS_PER_PACK];
  __128_BITS_MUT(pack[0]) = __128_BITS(input[idx]);
  auto thread_acc = ZERO<Acc>;
#pragma unroll
  for (auto i = 0; i < NUM_ELEMS_PER_PACK; i++) {
    const Acc val = (idx + i < n) ? static_cast<Acc>(pack[i])
                                  : ZERO<Acc>; // 读取输入数据，越界则为0
    thread_acc = thread_acc + val;
  }
  const Acc warp_acc = butterfly_reduce(thread_acc); // 计算warp内的规约和
  if (laneid == 0) {                                 // 每个warp的第一个线程将
    reduce_smem[warpid] = warp_acc;
  }
  __syncthreads(); // 同步所有线程，确保共享内存中的数据
  if (warpid == 0) {
    const Acc warp_acc =
        (laneid < (NUM_THREADS_PER_BLOCK + WARP_SIZE - 1) / WARP_SIZE)
            ? reduce_smem[laneid]
            : ZERO<Acc>;                              // 读取每个warp的部分和
    const Acc block_acc = butterfly_reduce(warp_acc); // 计算block内的规约和
    if (laneid == 0) {
      atomicAdd(output, block_acc); // 使用原子操作将结果累加到输出
    }
  }
}

template <typename Scalar, typename Vector, typename Acc = Scalar,
          const int NUM_ELEM_PER_BLOCK = 256>
  requires IsVectorOf<Vector, Scalar> && ThreadReduceable<Vector, Acc>
__global__ auto reduce_vector_kernel(const Scalar *__restrict__ input,
                                     Acc *__restrict__ output, int n) -> void {
  static_assert(NUM_ELEM_PER_BLOCK % VectorTraits<Vector>::SIZE == 0,
                "NUM_ELEM_PER_BLOCK must be multiple of Vector::SIZE");
  constexpr auto NUM_THREADS_PER_BLOCK =
      NUM_ELEM_PER_BLOCK / VectorTraits<Vector>::SIZE;
  constexpr auto NUM_WARPS_PER_BLOCK =
      (NUM_THREADS_PER_BLOCK + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ Acc reduce_smem[NUM_WARPS_PER_BLOCK];

  const auto tid = threadIdx.x;
  const auto bid = blockIdx.x;
  const auto idx = (blockDim.x * bid + tid) * VectorTraits<Vector>::SIZE;
  const Vector vec = (idx < n)
                         ? (reinterpret_cast<const Vector *>(&input[idx]))[0]
                         : ZERO<Vector>;
  const Acc thread_acc = thread_reduce<Vector, Acc>(vec);
  const auto warpid = threadIdx.x / WARP_SIZE;
  const auto laneid = threadIdx.x % WARP_SIZE;
  const Acc warp_acc = butterfly_reduce(thread_acc);
  if (laneid == 0) {
    reduce_smem[warpid] = warp_acc;
  }
  __syncthreads();
  if (warpid == 0) {
    const Acc warp_acc =
        (laneid < NUM_WARPS_PER_BLOCK) ? reduce_smem[laneid] : ZERO<Acc>;
    const Acc block_acc = butterfly_reduce(warp_acc);
    if (laneid == 0) {
      atomicAdd(output, block_acc);
    }
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  TORCH_CHECK(((T).options().dtype() == (th_type)), "values must "             \
                                                    "be " #th_type)

#define CHECK_CUDA(T) TORCH_CHECK((T).is_cuda(), "tensor must be a CUDA tensor")

#define LAUNCH_REDUCE_SCALAR_KERNEL(NT, grid, scalar_t, acc_t, x_ptr, y_ptr,   \
                                    n)                                         \
  do {                                                                         \
    dim3 block((NT));                                                          \
    reduce_scalar_kernel<scalar_t, acc_t, (NT)>                                \
        <<<(grid), block>>>((x_ptr), (y_ptr), (n));                            \
  } while (0)

#define DISPATCH_REDUCE_SCALAR_KERNEL(K, grid, scalar_t, acc_t, x_ptr, y_ptr,  \
                                      n)                                       \
  do {                                                                         \
    switch ((K)) {                                                             \
    case 32:                                                                   \
      LAUNCH_REDUCE_SCALAR_KERNEL(32, grid, scalar_t, acc_t, x_ptr, y_ptr, n); \
      break;                                                                   \
    case 64:                                                                   \
      LAUNCH_REDUCE_SCALAR_KERNEL(64, grid, scalar_t, acc_t, x_ptr, y_ptr, n); \
      break;                                                                   \
    case 128:                                                                  \
      LAUNCH_REDUCE_SCALAR_KERNEL(128, grid, scalar_t, acc_t, x_ptr, y_ptr,    \
                                  n);                                          \
      break;                                                                   \
    case 256:                                                                  \
      LAUNCH_REDUCE_SCALAR_KERNEL(256, grid, scalar_t, acc_t, x_ptr, y_ptr,    \
                                  n);                                          \
      break;                                                                   \
    case 512:                                                                  \
      LAUNCH_REDUCE_SCALAR_KERNEL(512, grid, scalar_t, acc_t, x_ptr, y_ptr,    \
                                  n);                                          \
      break;                                                                   \
    case 1024:                                                                 \
      LAUNCH_REDUCE_SCALAR_KERNEL(1024, grid, scalar_t, acc_t, x_ptr, y_ptr,   \
                                  n);                                          \
      break;                                                                   \
    default:                                                                   \
      throw std::runtime_error("only support K: 32/64/128/256/512/1024");      \
    }                                                                          \
  } while (0)

#define LAUNCH_REDUCE_VECTOR_KERNEL(NT, grid, scalar_t, vec_t, acc_t, x_ptr,   \
                                    y_ptr, n)                                  \
  do {                                                                         \
    constexpr int V = VectorTraits<vec_t>::SIZE;                               \
    dim3 block((NT) / V);                                                      \
    reduce_vector_kernel<scalar_t, vec_t, acc_t, (NT)>                         \
        <<<(grid), block>>>((x_ptr), (y_ptr), (n));                            \
  } while (0)

#define DISPATCH_REDUCE_VECTOR_KERNEL(K, grid, scalar_t, vec_t, acc_t, x_ptr,  \
                                      y_ptr, n)                                \
  do {                                                                         \
    switch ((K)) {                                                             \
    case 32:                                                                   \
      LAUNCH_REDUCE_VECTOR_KERNEL(32, grid, scalar_t, vec_t, acc_t, x_ptr,     \
                                  y_ptr, n);                                   \
      break;                                                                   \
    case 64:                                                                   \
      LAUNCH_REDUCE_VECTOR_KERNEL(64, grid, scalar_t, vec_t, acc_t, x_ptr,     \
                                  y_ptr, n);                                   \
      break;                                                                   \
    case 128:                                                                  \
      LAUNCH_REDUCE_VECTOR_KERNEL(128, grid, scalar_t, vec_t, acc_t, x_ptr,    \
                                  y_ptr, n);                                   \
      break;                                                                   \
    case 256:                                                                  \
      LAUNCH_REDUCE_VECTOR_KERNEL(256, grid, scalar_t, vec_t, acc_t, x_ptr,    \
                                  y_ptr, n);                                   \
      break;                                                                   \
    case 512:                                                                  \
      LAUNCH_REDUCE_VECTOR_KERNEL(512, grid, scalar_t, vec_t, acc_t, x_ptr,    \
                                  y_ptr, n);                                   \
      break;                                                                   \
    case 1024:                                                                 \
      LAUNCH_REDUCE_VECTOR_KERNEL(1024, grid, scalar_t, vec_t, acc_t, x_ptr,   \
                                  y_ptr, n);                                   \
      break;                                                                   \
    default:                                                                   \
      throw std::runtime_error("only support K: 32/64/128/256/512/1024");      \
    }                                                                          \
  } while (0)

#define LAUNCH_REDUCE_PACK_KERNEL(NT, grid, scalar_t, acc_t, pack_elems,       \
                                  x_ptr, y_ptr, n)                             \
  do {                                                                         \
    dim3 block((NT) / (pack_elems));                                           \
    reduce_pack_kernel<scalar_t, acc_t, (NT)>                                  \
        <<<(grid), block>>>((x_ptr), (y_ptr), (n));                            \
  } while (0)

#define DISPATCH_REDUCE_PACK_KERNEL(K, grid, scalar_t, acc_t, pack_elems,      \
                                    x_ptr, y_ptr, n)                           \
  do {                                                                         \
    switch ((K)) {                                                             \
    case 32:                                                                   \
      LAUNCH_REDUCE_PACK_KERNEL(32, grid, scalar_t, acc_t, pack_elems, x_ptr,  \
                                y_ptr, n);                                     \
      break;                                                                   \
    case 64:                                                                   \
      LAUNCH_REDUCE_PACK_KERNEL(64, grid, scalar_t, acc_t, pack_elems, x_ptr,  \
                                y_ptr, n);                                     \
      break;                                                                   \
    case 128:                                                                  \
      LAUNCH_REDUCE_PACK_KERNEL(128, grid, scalar_t, acc_t, pack_elems, x_ptr, \
                                y_ptr, n);                                     \
      break;                                                                   \
    case 256:                                                                  \
      LAUNCH_REDUCE_PACK_KERNEL(256, grid, scalar_t, acc_t, pack_elems, x_ptr, \
                                y_ptr, n);                                     \
      break;                                                                   \
    case 512:                                                                  \
      LAUNCH_REDUCE_PACK_KERNEL(512, grid, scalar_t, acc_t, pack_elems, x_ptr, \
                                y_ptr, n);                                     \
      break;                                                                   \
    case 1024:                                                                 \
      LAUNCH_REDUCE_PACK_KERNEL(1024, grid, scalar_t, acc_t, pack_elems,       \
                                x_ptr, y_ptr, n);                              \
      break;                                                                   \
    default:                                                                   \
      throw std::runtime_error("only support K: 32/64/128/256/512/1024");      \
    }                                                                          \
  } while (0)

#define TORCH_BINDING_REDUCE_SCALAR(scalar_tag, acc_tag, th_type, scalar_t,    \
                                    acc_t)                                     \
  torch::Tensor reduce_scalar_sum_##scalar_tag##_##acc_tag(torch::Tensor x) {  \
    CHECK_CUDA(x)                                                              \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    auto x_contig = x.contiguous();                                            \
    auto y = torch::zeros({1}, x_contig.options().dtype(torch::kFloat32));     \
    const int64_t n64 = x_contig.numel();                                      \
    const int n = static_cast<int>(n64);                                       \
    if (n <= 0) {                                                              \
      return y;                                                                \
    }                                                                          \
    const int ndim = x_contig.dim();                                           \
    if (ndim == 2) {                                                           \
      const int S = static_cast<int>(x_contig.size(0));                        \
      const int K = static_cast<int>(x_contig.size(1));                        \
      if (K <= 1024) {                                                         \
        dim3 grid(S);                                                          \
        auto x_ptr = reinterpret_cast<const scalar_t *>(x_contig.data_ptr());  \
        auto y_ptr = reinterpret_cast<acc_t *>(y.data_ptr());                  \
        DISPATCH_REDUCE_SCALAR_KERNEL(K, grid, scalar_t, acc_t, x_ptr, y_ptr,  \
                                      n);                                      \
        return y;                                                              \
      }                                                                        \
    }                                                                          \
    constexpr int EPB = 1024;                                                  \
    dim3 block(EPB);                                                           \
    dim3 grid((n + EPB - 1) / EPB);                                            \
    reduce_scalar_kernel<scalar_t, acc_t, EPB><<<grid, block>>>(               \
        reinterpret_cast<const scalar_t *>(x_contig.data_ptr()),               \
        reinterpret_cast<acc_t *>(y.data_ptr()), n);                           \
    return y;                                                                  \
  }

#define TORCH_BINDING_REDUCE_VECTOR(vec_tag, acc_tag, th_type, scalar_t,       \
                                    vec_t, acc_t)                              \
  torch::Tensor reduce_vector_sum_##vec_tag##_##acc_tag(torch::Tensor x) {     \
    static_assert(IsVectorOf<vec_t, scalar_t>,                                 \
                  "IsVectorOf failed for " STRINGFY(vec_tag));                 \
    static_assert(ThreadReduceable<vec_t, acc_t>,                              \
                  "ThreadReduceable failed for " STRINGFY(vec_tag));           \
    CHECK_CUDA(x)                                                              \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    auto x_contig = x.contiguous();                                            \
    auto y = torch::zeros({1}, x_contig.options().dtype(torch::kFloat32));     \
    const int64_t n64 = x_contig.numel();                                      \
    const int n = static_cast<int>(n64);                                       \
    if (n <= 0) {                                                              \
      return y;                                                                \
    }                                                                          \
    const int ndim = x_contig.dim();                                           \
    if (ndim == 2) {                                                           \
      const int S = static_cast<int>(x_contig.size(0));                        \
      const int K = static_cast<int>(x_contig.size(1));                        \
      constexpr int V = VectorTraits<vec_t>::SIZE;                             \
      if (K <= 1024 && (K % V) == 0) {                                         \
        dim3 grid(S);                                                          \
        auto x_ptr = reinterpret_cast<const scalar_t *>(x_contig.data_ptr());  \
        auto y_ptr = reinterpret_cast<acc_t *>(y.data_ptr());                  \
        DISPATCH_REDUCE_VECTOR_KERNEL(K, grid, scalar_t, vec_t, acc_t, x_ptr,  \
                                      y_ptr, n);                               \
        return y;                                                              \
      }                                                                        \
    }                                                                          \
    constexpr int EPB = 1024;                                                  \
    dim3 block(EPB);                                                           \
    dim3 grid((n + EPB - 1) / EPB);                                            \
    reduce_scalar_kernel<scalar_t, acc_t, EPB><<<grid, block>>>(               \
        reinterpret_cast<const scalar_t *>(x_contig.data_ptr()),               \
        reinterpret_cast<acc_t *>(y.data_ptr()), n);                           \
    return y;                                                                  \
  }

#define TORCH_BINDING_REDUCE_PACK(scalar_tag, acc_tag, th_type, scalar_t,      \
                                  acc_t)                                       \
  torch::Tensor reduce_pack_sum_##scalar_tag##_##acc_tag(torch::Tensor x) {    \
    CHECK_CUDA(x)                                                              \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    auto x_contig = x.contiguous();                                            \
    auto y = torch::zeros({1}, x_contig.options().dtype(torch::kFloat32));     \
    const int64_t n64 = x_contig.numel();                                      \
    const int n = static_cast<int>(n64);                                       \
    if (n <= 0) {                                                              \
      return y;                                                                \
    }                                                                          \
    const int ndim = x_contig.dim();                                           \
    constexpr int PACK_ELEMS = 16 / static_cast<int>(sizeof(scalar_t));        \
    static_assert(PACK_ELEMS > 0, "PACK_ELEMS must be positive");              \
    if (ndim == 2) {                                                           \
      const int S = static_cast<int>(x_contig.size(0));                        \
      const int K = static_cast<int>(x_contig.size(1));                        \
      if (K <= 1024 && (K % PACK_ELEMS) == 0) {                                \
        dim3 grid(S);                                                          \
        auto x_ptr = reinterpret_cast<const scalar_t *>(x_contig.data_ptr());  \
        auto y_ptr = reinterpret_cast<acc_t *>(y.data_ptr());                  \
        DISPATCH_REDUCE_PACK_KERNEL(K, grid, scalar_t, acc_t, PACK_ELEMS,      \
                                    x_ptr, y_ptr, n);                          \
        return y;                                                              \
      }                                                                        \
    }                                                                          \
    constexpr int EPB = 1024;                                                  \
    dim3 block(EPB);                                                           \
    dim3 grid((n + EPB - 1) / EPB);                                            \
    reduce_scalar_kernel<scalar_t, acc_t, EPB><<<grid, block>>>(               \
        reinterpret_cast<const scalar_t *>(x_contig.data_ptr()),               \
        reinterpret_cast<acc_t *>(y.data_ptr()), n);                           \
    return y;                                                                  \
  }

TORCH_BINDING_REDUCE_SCALAR(f16, f32, torch::kHalf, f16, f32)
TORCH_BINDING_REDUCE_SCALAR(bf16, f32, torch::kBFloat16, bf16, f32)

TORCH_BINDING_REDUCE_VECTOR(f16x2, f32, torch::kHalf, f16, f16x2, f32)
TORCH_BINDING_REDUCE_VECTOR(bf16x2, f32, torch::kBFloat16, bf16, bf16x2, f32)

TORCH_BINDING_REDUCE_PACK(f32, f32, torch::kFloat32, f32, f32)
TORCH_BINDING_REDUCE_PACK(bf16, f32, torch::kBFloat16, bf16, f32)

// f32
TORCH_BINDING_REDUCE_SCALAR(f32, f32, torch::kFloat32, f32, f32)
TORCH_BINDING_REDUCE_VECTOR(f32x4, f32, torch::kFloat32, f32, f32x4, f32)

// f16
TORCH_BINDING_REDUCE_PACK(f16, f16, torch::kHalf, f16, f16)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(reduce_scalar_sum_f32_f32)
  TORCH_BINDING_COMMON_EXTENSION(reduce_scalar_sum_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(reduce_scalar_sum_bf16_f32)

  TORCH_BINDING_COMMON_EXTENSION(reduce_vector_sum_f32x4_f32)
  TORCH_BINDING_COMMON_EXTENSION(reduce_vector_sum_f16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(reduce_vector_sum_bf16x2_f32)

  TORCH_BINDING_COMMON_EXTENSION(reduce_pack_sum_f32_f32)
  TORCH_BINDING_COMMON_EXTENSION(reduce_pack_sum_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(reduce_pack_sum_bf16_f32)
}