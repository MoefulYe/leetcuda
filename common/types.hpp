#include <concepts>
#include <cstdint>
#include <cuda/std/limits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <sys/stat.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/types.h>
#include <type_traits>
#include <vector_types.h>

namespace leetcuda {
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using f8e5m2 = __nv_fp8_e5m2;
using f8e4m3 = __nv_fp8_e4m3;
using f16 = half;
using f32 = float;
using f64 = double;
using bf16 = nv_bfloat16;

using f8e4m3x2 = __nv_fp8x2_e4m3;
using f8e4m3x4 = __nv_fp8x4_e4m3;
using f8e5m2x2 = __nv_fp8x2_e5m2;
using f8e5m2x4 = __nv_fp8x4_e5m2;
using f16x2 = half2;
using f32x2 = float2;
using f32x4 = float4;
using f64x2 = double2;
using bf16x2 = nv_bfloat162;
using i8x2 = char2;
using i8x4 = char4;
using u8x2 = uchar2;
using u8x4 = uchar4;
using i16x2 = short2;
using i16x4 = short4;
using u16x2 = ushort2;
using u16x4 = ushort4;
using i32x2 = int2;
using i32x4 = int4;
using u32x2 = uint2;
using u32x4 = uint4;
using i64x2 = longlong2;
using i64x4 = longlong4;
using u64x2 = ulonglong2;
using u64x4 = ulonglong4;

using __128_bits = float4;

#define INT4(lvalue) (reinterpret_cast<const int4 *>(&(lvalue))[0])
#define F32X4(lvalue) (reinterpret_cast<const f32x4 *>(&(lvalue))[0])
#define F16X2(lvalue) (reinterpret_cast<const f16x2 *>(&(lvalue))[0])
#define BF16X2(lvalue) (reinterpret_cast<const b16x2 *>(&(lvalue))[0])
#define F32X4(lvalue) (reinterpret_cast<const f32x4 *>(&(lvalue))[0])
#define __128_BITS(lvalue) (reinterpret_cast<const __128_bits *>(&(lvalue))[0])

#define INT4_MUT(lvalue) (reinterpret_cast<int4 *>(&(lvalue))[0])
#define F32X4_MUT(lvalue) (reinterpret_cast<f32x4 *>(&(lvalue))[0])
#define F16X2_MUT(lvalue) (reinterpret_cast<f16x2 *>(&(lvalue))[0])
#define BF16X2_MUT(lvalue) (reinterpret_cast<b16x2 *>(&(lvalue))[0])
#define F32X4_MUT(lvalue) (reinterpret_cast<f32x4 *>(&(lvalue))[0])
#define __128_BITS_MUT(lvalue) (reinterpret_cast<__128_bits *>(&(lvalue))[0])

template <typename T>
__device__ __host__ __forceinline__ auto zero() -> T = delete;

template <typename T>
__device__ __host__ __forceinline__ auto lowest() -> T = delete;

template <typename T>
__device__ __host__ __forceinline__ auto greatest() -> T = delete;

template <> __device__ __host__ __forceinline__ auto zero<f32>() -> f32 {
  return 0.0f;
}

template <> __device__ __host__ __forceinline__ auto lowest<f32>() -> f32 {
  return cuda::std::numeric_limits<f32>::lowest();
}

template <> __device__ __host__ __forceinline__ auto greatest<f32>() -> f32 {
  return cuda::std::numeric_limits<f32>::max();
}

template <> __device__ __host__ __forceinline__ auto zero<f64>() -> f64 {
  return 0.0;
}

template <> __device__ __host__ __forceinline__ auto lowest<f64>() -> f64 {
  return cuda::std::numeric_limits<f64>::lowest();
}

template <> __device__ __host__ __forceinline__ auto greatest<f64>() -> f64 {
  return cuda::std::numeric_limits<f64>::max();
}

template <> __device__ __host__ __forceinline__ auto zero<i8>() -> i8 {
  return static_cast<i8>(0);
}

template <> __device__ __host__ __forceinline__ auto lowest<i8>() -> i8 {
  return cuda::std::numeric_limits<i8>::lowest();
}

template <> __device__ __host__ __forceinline__ auto greatest<i8>() -> i8 {
  return cuda::std::numeric_limits<i8>::max();
}

template <> __device__ __host__ __forceinline__ auto zero<i16>() -> i16 {
  return static_cast<i16>(0);
}

template <> __device__ __host__ __forceinline__ auto lowest<i16>() -> i16 {
  return cuda::std::numeric_limits<i16>::lowest();
}

template <> __device__ __host__ __forceinline__ auto greatest<i16>() -> i16 {
  return cuda::std::numeric_limits<i16>::max();
}

template <> __device__ __host__ __forceinline__ auto zero<i32>() -> i32 {
  return static_cast<i32>(0);
}

template <> __device__ __host__ __forceinline__ auto lowest<i32>() -> i32 {
  return cuda::std::numeric_limits<i32>::lowest();
}

template <> __device__ __host__ __forceinline__ auto greatest<i32>() -> i32 {
  return cuda::std::numeric_limits<i32>::max();
}

template <> __device__ __host__ __forceinline__ auto zero<i64>() -> i64 {
  return static_cast<i64>(0);
}

template <> __device__ __host__ __forceinline__ auto lowest<i64>() -> i64 {
  return cuda::std::numeric_limits<i64>::lowest();
}

template <> __device__ __host__ __forceinline__ auto greatest<i64>() -> i64 {
  return cuda::std::numeric_limits<i64>::max();
}

template <> __device__ __host__ __forceinline__ auto zero<u8>() -> u8 {
  return static_cast<u8>(0);
}

template <> __device__ __host__ __forceinline__ auto lowest<u8>() -> u8 {
  return cuda::std::numeric_limits<u8>::lowest();
}

template <> __device__ __host__ __forceinline__ auto greatest<u8>() -> u8 {
  return cuda::std::numeric_limits<u8>::max();
}

template <> __device__ __host__ __forceinline__ auto zero<u16>() -> u16 {
  return static_cast<u16>(0);
}

template <> __device__ __host__ __forceinline__ auto lowest<u16>() -> u16 {
  return cuda::std::numeric_limits<u16>::lowest();
}

template <> __device__ __host__ __forceinline__ auto greatest<u16>() -> u16 {
  return cuda::std::numeric_limits<u16>::max();
}

template <> __device__ __host__ __forceinline__ auto zero<u32>() -> u32 {
  return static_cast<u32>(0);
}

template <> __device__ __host__ __forceinline__ auto lowest<u32>() -> u32 {
  return cuda::std::numeric_limits<u32>::lowest();
}

template <> __device__ __host__ __forceinline__ auto greatest<u32>() -> u32 {
  return cuda::std::numeric_limits<u32>::max();
}

template <> __device__ __host__ __forceinline__ auto zero<u64>() -> u64 {
  return static_cast<u64>(0);
}

template <> __device__ __host__ __forceinline__ auto lowest<u64>() -> u64 {
  return cuda::std::numeric_limits<u64>::lowest();
}

template <> __device__ __host__ __forceinline__ auto greatest<u64>() -> u64 {
  return cuda::std::numeric_limits<u64>::max();
}

template <> __device__ __host__ __forceinline__ auto zero<f16>() -> f16 {
  return f16{};
}

template <> __device__ __host__ __forceinline__ auto lowest<f16>() -> f16 {
  return cuda::std::numeric_limits<f16>::lowest();
}

template <> __device__ __host__ __forceinline__ auto greatest<f16>() -> f16 {
  return cuda::std::numeric_limits<f16>::max();
}

template <> __device__ __host__ __forceinline__ auto zero<bf16>() -> bf16 {
  return bf16{};
}

template <> __device__ __host__ __forceinline__ auto lowest<bf16>() -> bf16 {
  return cuda::std::numeric_limits<bf16>::lowest();
}

template <> __device__ __host__ __forceinline__ auto greatest<bf16>() -> bf16 {
  return cuda::std::numeric_limits<bf16>::max();
}

template <> __device__ __host__ __forceinline__ auto zero<f8e4m3>() -> f8e4m3 {
  return f8e4m3{};
}

template <>
__device__ __host__ __forceinline__ auto lowest<f8e4m3>() -> f8e4m3 {
  return cuda::std::numeric_limits<f8e4m3>::lowest();
}

template <>
__device__ __host__ __forceinline__ auto greatest<f8e4m3>() -> f8e4m3 {
  return cuda::std::numeric_limits<f8e4m3>::max();
}

template <> __device__ __host__ __forceinline__ auto zero<f8e5m2>() -> f8e5m2 {
  return f8e5m2{};
}

template <>
__device__ __host__ __forceinline__ auto lowest<f8e5m2>() -> f8e5m2 {
  return cuda::std::numeric_limits<f8e5m2>::lowest();
}

template <>
__device__ __host__ __forceinline__ auto greatest<f8e5m2>() -> f8e5m2 {
  return cuda::std::numeric_limits<f8e5m2>::max();
}

template <>
__device__ __host__ __forceinline__ auto zero<f8e4m3x2>() -> f8e4m3x2 {
  return f8e4m3x2{};
}

template <>
__device__ __host__ __forceinline__ auto zero<f8e4m3x4>() -> f8e4m3x4 {
  return f8e4m3x4{};
}

template <>
__device__ __host__ __forceinline__ auto zero<f8e5m2x2>() -> f8e5m2x2 {
  return f8e5m2x2{};
}

template <>
__device__ __host__ __forceinline__ auto zero<f8e5m2x4>() -> f8e5m2x4 {
  return f8e5m2x4{};
}

template <> __device__ __host__ __forceinline__ auto zero<f16x2>() -> f16x2 {
  return f16x2{};
}

template <> __device__ __host__ __forceinline__ auto zero<bf16x2>() -> bf16x2 {
  return bf16x2{};
}

template <> __device__ __host__ __forceinline__ auto zero<f32x2>() -> f32x2 {
  return f32x2{0.0f, 0.0f};
}

template <> __device__ __host__ __forceinline__ auto zero<f32x4>() -> f32x4 {
  return f32x4{0.0f, 0.0f, 0.0f, 0.0f};
}

template <> __device__ __host__ __forceinline__ auto zero<f64x2>() -> f64x2 {
  return f64x2{0.0, 0.0};
}

template <> __device__ __host__ __forceinline__ auto zero<i8x2>() -> i8x2 {
  return i8x2{0, 0};
}

template <> __device__ __host__ __forceinline__ auto zero<i8x4>() -> i8x4 {
  return i8x4{0, 0, 0, 0};
}

template <> __device__ __host__ __forceinline__ auto zero<u8x2>() -> u8x2 {
  return u8x2{0, 0};
}

template <> __device__ __host__ __forceinline__ auto zero<u8x4>() -> u8x4 {
  return u8x4{0, 0, 0, 0};
}

template <> __device__ __host__ __forceinline__ auto zero<i16x2>() -> i16x2 {
  return i16x2{0, 0};
}

template <> __device__ __host__ __forceinline__ auto zero<i16x4>() -> i16x4 {
  return i16x4{0, 0, 0, 0};
}

template <> __device__ __host__ __forceinline__ auto zero<u16x2>() -> u16x2 {
  return u16x2{0, 0};
}

template <> __device__ __host__ __forceinline__ auto zero<u16x4>() -> u16x4 {
  return u16x4{0, 0, 0, 0};
}

template <> __device__ __host__ __forceinline__ auto zero<i32x2>() -> i32x2 {
  return i32x2{0, 0};
}

template <> __device__ __host__ __forceinline__ auto zero<i32x4>() -> i32x4 {
  return i32x4{0, 0, 0, 0};
}

template <> __device__ __host__ __forceinline__ auto zero<u32x2>() -> u32x2 {
  return u32x2{0u, 0u};
}

template <> __device__ __host__ __forceinline__ auto zero<u32x4>() -> u32x4 {
  return u32x4{0u, 0u, 0u, 0u};
}

template <> __device__ __host__ __forceinline__ auto zero<i64x2>() -> i64x2 {
  return i64x2{0ll, 0ll};
}

template <> __device__ __host__ __forceinline__ auto zero<i64x4>() -> i64x4 {
  return i64x4{0ll, 0ll, 0ll, 0ll};
}

template <> __device__ __host__ __forceinline__ auto zero<u64x2>() -> u64x2 {
  return u64x2{0ull, 0ull};
}

template <> __device__ __host__ __forceinline__ auto zero<u64x4>() -> u64x4 {
  return u64x4{0ull, 0ull, 0ull, 0ull};
}

template <typename T> struct _VectorTraits {
  // 默认大小为 0，表示这不是一个向量
  static constexpr int SIZE = 0;

  // 默认类型为 void，用于后续 SFINAE 检查
  using Scalar = void;
};

template <> struct _VectorTraits<f8e4m3x4> {
  static constexpr int SIZE = 4;
  using Scalar = f8e4m3;
};

template <> struct _VectorTraits<f8e4m3x2> {
  static constexpr int SIZE = 2;
  using Scalar = f8e4m3;
};

template <> struct _VectorTraits<f8e5m2x4> {
  static constexpr int SIZE = 4;
  using Scalar = f8e5m2;
};

template <> struct _VectorTraits<f8e5m2x2> {
  static constexpr int SIZE = 2;
  using Scalar = f8e5m2;
};

template <> struct _VectorTraits<f32x4> {
  static constexpr int SIZE = 4;
  using Scalar = f32;
};

template <> struct _VectorTraits<f32x2> {
  static constexpr int SIZE = 2;
  using Scalar = f32;
};

template <> struct _VectorTraits<f16x2> {
  static constexpr int SIZE = 2;
  using Scalar = f16;
};

template <> struct _VectorTraits<bf16x2> {
  static constexpr int SIZE = 2;
  using Scalar = bf16;
};

template <> struct _VectorTraits<f64x2> {
  static constexpr int SIZE = 2;
  using Scalar = f64;
};

template <> struct _VectorTraits<i8x4> {
  static constexpr int SIZE = 4;
  using Scalar = i8;
};

template <> struct _VectorTraits<i8x2> {
  static constexpr int SIZE = 2;
  using Scalar = i8;
};

template <> struct _VectorTraits<u8x4> {
  static constexpr int SIZE = 4;
  using Scalar = u8;
};

template <> struct _VectorTraits<u8x2> {
  static constexpr int SIZE = 2;
  using Scalar = u8;
};

template <> struct _VectorTraits<i16x4> {
  static constexpr int SIZE = 4;
  using Scalar = i16;
};

template <> struct _VectorTraits<i16x2> {
  static constexpr int SIZE = 2;
  using Scalar = i16;
};

template <> struct _VectorTraits<u16x4> {
  static constexpr int SIZE = 4;
  using Scalar = u16;
};

template <> struct _VectorTraits<u16x2> {
  static constexpr int SIZE = 2;
  using Scalar = u16;
};

template <> struct _VectorTraits<i32x4> {
  static constexpr int SIZE = 4;
  using Scalar = i32;
};

template <> struct _VectorTraits<i32x2> {
  static constexpr int SIZE = 2;
  using Scalar = i32;
};

template <> struct _VectorTraits<u32x4> {
  static constexpr int SIZE = 4;
  using Scalar = u32;
};

template <> struct _VectorTraits<u32x2> {
  static constexpr int SIZE = 2;
  using Scalar = u32;
};

template <> struct _VectorTraits<i64x4> {
  static constexpr int SIZE = 4;
  using Scalar = i64;
};

template <> struct _VectorTraits<i64x2> {
  static constexpr int SIZE = 2;
  using Scalar = i64;
};

template <> struct _VectorTraits<u64x4> {
  static constexpr int SIZE = 4;
  using Scalar = u64;
};

template <> struct _VectorTraits<u64x2> {
  static constexpr int SIZE = 2;
  using Scalar = u64;
};

template <typename Vector>
concept IsVector = requires {
  typename _VectorTraits<Vector>::Scalar;
  { _VectorTraits<Vector>::SIZE } -> std::convertible_to<int>;
};

static_assert(IsVector<f8e4m3x4>);
static_assert(IsVector<f8e4m3x2>);
static_assert(IsVector<f8e5m2x4>);
static_assert(IsVector<f8e5m2x2>);
static_assert(IsVector<f32x4>);
static_assert(IsVector<f32x2>);
static_assert(IsVector<f16x2>);
static_assert(IsVector<bf16x2>);
static_assert(IsVector<f64x2>);
static_assert(IsVector<i8x4>);
static_assert(IsVector<i8x2>);
static_assert(IsVector<u8x4>);
static_assert(IsVector<u8x2>);
static_assert(IsVector<i16x4>);
static_assert(IsVector<i16x2>);
static_assert(IsVector<u16x4>);
static_assert(IsVector<u16x2>);
static_assert(IsVector<i32x4>);
static_assert(IsVector<i32x2>);
static_assert(IsVector<u32x4>);
static_assert(IsVector<u32x2>);
static_assert(IsVector<i64x4>);
static_assert(IsVector<i64x2>);
static_assert(IsVector<u64x4>);
static_assert(IsVector<u64x2>);

template <typename Vector>
using ScalarOf = typename _VectorTraits<Vector>::Scalar;

template <typename Vector>
constexpr auto NUM_OF_ELEMENTS = _VectorTraits<Vector>::SIZE;

template <IsVector Vector>
__device__ __host__ __forceinline__ auto lowest() -> Vector {
  using Scalar = ScalarOf<Vector>;
  if constexpr (NUM_OF_ELEMENTS<Vector> == 2) {
    return Vector{lowest<Scalar>(), lowest<Scalar>()};
  } else if constexpr (NUM_OF_ELEMENTS<Vector> == 4) {
    return Vector{lowest<Scalar>(), lowest<Scalar>(), lowest<Scalar>(),
                  lowest<Scalar>()};
  } else {
    return Vector{};
  }
}

template <IsVector Vector>
__device__ __host__ __forceinline__ auto greatest() -> Vector {
  using Scalar = ScalarOf<Vector>;
  if constexpr (NUM_OF_ELEMENTS<Vector> == 2) {
    return Vector{greatest<Scalar>(), greatest<Scalar>()};
  } else if constexpr (NUM_OF_ELEMENTS<Vector> == 4) {
    return Vector{greatest<Scalar>(), greatest<Scalar>(), greatest<Scalar>(),
                  greatest<Scalar>()};
  } else {
    return Vector{};
  }
}

template <typename Scalar, int N> struct _Vector {
  using Type = void;
};

template <typename Scalar, int N>
using VectorOf = typename _Vector<Scalar, N>::Type;

template <typename Vector> using VectorTraits = _VectorTraits<Vector>;

template <typename Vector, typename Scalar>
inline constexpr bool IsVectorOf =
    IsVector<Vector> && std::is_same_v<ScalarOf<Vector>, Scalar>;

template <> struct _Vector<f8e4m3, 2> {
  using Type = f8e4m3x2;
};

template <> struct _Vector<f8e4m3, 4> {
  using Type = f8e4m3x4;
};

template <> struct _Vector<f8e5m2, 2> {
  using Type = f8e5m2x2;
};

template <> struct _Vector<f8e5m2, 4> {
  using Type = f8e5m2x4;
};

template <> struct _Vector<f16, 2> {
  using Type = f16x2;
};

template <> struct _Vector<bf16, 2> {
  using Type = bf16x2;
};

template <> struct _Vector<f32, 2> {
  using Type = f32x2;
};

template <> struct _Vector<f32, 4> {
  using Type = f32x4;
};

template <> struct _Vector<f64, 2> {
  using Type = f64x2;
};

template <> struct _Vector<i8, 2> {
  using Type = i8x2;
};

template <> struct _Vector<i8, 4> {
  using Type = i8x4;
};

template <> struct _Vector<u8, 2> {
  using Type = u8x2;
};

template <> struct _Vector<u8, 4> {
  using Type = u8x4;
};

template <> struct _Vector<i16, 2> {
  using Type = i16x2;
};

template <> struct _Vector<i16, 4> {
  using Type = i16x4;
};

template <> struct _Vector<u16, 2> {
  using Type = u16x2;
};

template <> struct _Vector<u16, 4> {
  using Type = u16x4;
};

template <> struct _Vector<i32, 2> {
  using Type = i32x2;
};

template <> struct _Vector<i32, 4> {
  using Type = i32x4;
};

template <> struct _Vector<u32, 2> {
  using Type = u32x2;
};

template <> struct _Vector<u32, 4> {
  using Type = u32x4;
};

template <> struct _Vector<i64, 2> {
  using Type = i64x2;
};

template <> struct _Vector<i64, 4> {
  using Type = i64x4;
};

template <> struct _Vector<u64, 2> {
  using Type = u64x2;
};

template <> struct _Vector<u64, 4> {
  using Type = u64x4;
};

template <typename ScalarType>
constexpr auto cuda_to_torch_type() -> torch::ScalarType = delete;

template <> constexpr auto cuda_to_torch_type<f8e4m3>() -> torch::ScalarType {
  return torch::kFloat8_e4m3fn;
}

template <> constexpr auto cuda_to_torch_type<f8e5m2>() -> torch::ScalarType {
  return torch::kFloat8_e5m2;
}

template <> constexpr auto cuda_to_torch_type<f16>() -> torch::ScalarType {
  return torch::kFloat16;
}

template <> constexpr auto cuda_to_torch_type<bf16>() -> torch::ScalarType {
  return torch::kBFloat16;
}

template <> constexpr auto cuda_to_torch_type<f32>() -> torch::ScalarType {
  return torch::kFloat32;
}

template <> constexpr auto cuda_to_torch_type<f64>() -> torch::ScalarType {
  return torch::kFloat64;
}

template <> constexpr auto cuda_to_torch_type<i8>() -> torch::ScalarType {
  return torch::kInt8;
}

template <> constexpr auto cuda_to_torch_type<i16>() -> torch::ScalarType {
  return torch::kInt16;
}

template <> constexpr auto cuda_to_torch_type<i32>() -> torch::ScalarType {
  return torch::kInt32;
}

template <> constexpr auto cuda_to_torch_type<i64>() -> torch::ScalarType {
  return torch::kInt64;
}

template <> constexpr auto cuda_to_torch_type<u8>() -> torch::ScalarType {
  return torch::kUInt8;
}
template <> constexpr auto cuda_to_torch_type<u16>() -> torch::ScalarType {
  return torch::kUInt16;
}

template <> constexpr auto cuda_to_torch_type<u32>() -> torch::ScalarType {
  return torch::kUInt32;
}

template <> constexpr auto cuda_to_torch_type<u64>() -> torch::ScalarType {
  return torch::kUInt64;
}

template <typename T> struct Packed128Bits {
  static constexpr auto SIZE_BYTES = 128 / 8;
  static_assert(sizeof(T) < SIZE_BYTES, "Size of T must be less than 128 bits");
  static constexpr auto NUM_ELEMENTS = SIZE_BYTES / sizeof(T);
  static_assert(SIZE_BYTES % sizeof(T) == 0, "T must evenly divide 128 bits");
  T data[NUM_ELEMENTS];
  static auto __device__ __host__ __forceinline__ load(const void *ptr)
      -> Packed128Bits<T> {
    Packed128Bits<T> result;
    __128_BITS_MUT(result) =
        __128_BITS(*reinterpret_cast<const __128_bits *>(ptr));
    return result;
  }

  auto __device__ __host__ __forceinline__ store(void *ptr) const -> void {
    *reinterpret_cast<__128_bits *>(ptr) = __128_BITS(data);
  }
};
}; // namespace leetcuda