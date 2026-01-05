#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <vector_types.h>

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

template <typename T> inline constexpr auto ZERO = T{};
template <> inline constexpr f32 ZERO<f32> = 0.0f;
template <> inline constexpr f64 ZERO<f64> = 0.0;

template <> inline constexpr f16 ZERO<f16> = {};
template <> inline constexpr bf16 ZERO<bf16> = {};

template <> inline constexpr f32x4 ZERO<f32x4> = {0.0f, 0.0f, 0.0f, 0.0f};
template <> inline constexpr f32x2 ZERO<f32x2> = {0.0f, 0.0f};

template <> inline constexpr f16x2 ZERO<f16x2> = {};
template <> inline constexpr bf16x2 ZERO<bf16x2> = {};
template <> inline constexpr f64x2 ZERO<f64x2> = {0.0, 0.0};

template <typename Pack> inline constexpr auto PackSize = 0;

template <typename T> struct VectorTraits {
  // 默认大小为 0，表示这不是一个向量
  static constexpr int SIZE = 0;

  // 默认类型为 void，用于后续 SFINAE 检查
  using Scalar = void;
};

template <> struct VectorTraits<f8e4m3x4> {
  static constexpr int SIZE = 4;
  using Scalar = f8e4m3;
};

template <> struct VectorTraits<f8e4m3x2> {
  static constexpr int SIZE = 2;
  using Scalar = f8e4m3;
};

template <> struct VectorTraits<f8e5m2x4> {
  static constexpr int SIZE = 4;
  using Scalar = f8e5m2;
};

template <> struct VectorTraits<f8e5m2x2> {
  static constexpr int SIZE = 2;
  using Scalar = f8e5m2;
};

template <> struct VectorTraits<f32x4> {
  static constexpr int SIZE = 4;
  using Scalar = f32;
};

template <> struct VectorTraits<f32x2> {
  static constexpr int SIZE = 2;
  using Scalar = f32;
};

template <> struct VectorTraits<f16x2> {
  static constexpr int SIZE = 2;
  using Scalar = f16;
};

template <> struct VectorTraits<bf16x2> {
  static constexpr int SIZE = 2;
  using Scalar = bf16;
};

template <> struct VectorTraits<f64x2> {
  static constexpr int SIZE = 2;
  using Scalar = f64;
};

template <> struct VectorTraits<i8x4> {
  static constexpr int SIZE = 4;
  using Scalar = i8;
};

template <> struct VectorTraits<i8x2> {
  static constexpr int SIZE = 2;
  using Scalar = i8;
};

template <> struct VectorTraits<u8x4> {
  static constexpr int SIZE = 4;
  using Scalar = u8;
};

template <> struct VectorTraits<u8x2> {
  static constexpr int SIZE = 2;
  using Scalar = u8;
};

template <> struct VectorTraits<i16x4> {
  static constexpr int SIZE = 4;
  using Scalar = i16;
};

template <> struct VectorTraits<i16x2> {
  static constexpr int SIZE = 2;
  using Scalar = i16;
};

template <> struct VectorTraits<u16x4> {
  static constexpr int SIZE = 4;
  using Scalar = u16;
};

template <> struct VectorTraits<u16x2> {
  static constexpr int SIZE = 2;
  using Scalar = u16;
};

template <> struct VectorTraits<i32x4> {
  static constexpr int SIZE = 4;
  using Scalar = i32;
};

template <> struct VectorTraits<i32x2> {
  static constexpr int SIZE = 2;
  using Scalar = i32;
};

template <> struct VectorTraits<u32x4> {
  static constexpr int SIZE = 4;
  using Scalar = u32;
};

template <> struct VectorTraits<u32x2> {
  static constexpr int SIZE = 2;
  using Scalar = u32;
};

template <> struct VectorTraits<i64x4> {
  static constexpr int SIZE = 4;
  using Scalar = i64;
};

template <> struct VectorTraits<i64x2> {
  static constexpr int SIZE = 2;
  using Scalar = i64;
};

template <> struct VectorTraits<u64x4> {
  static constexpr int SIZE = 4;
  using Scalar = u64;
};

template <> struct VectorTraits<u64x2> {
  static constexpr int SIZE = 2;
  using Scalar = u64;
};

template <typename Vector, typename Scalar>
concept IsVectorOf = requires {
  typename VectorTraits<Vector>::Scalar;
  { VectorTraits<Vector>::SIZE } -> std::convertible_to<int>;
  requires std::is_same_v<typename VectorTraits<Vector>::Scalar, Scalar>;
};

static_assert(IsVectorOf<f8e4m3x4, f8e4m3>);
static_assert(IsVectorOf<f8e4m3x2, f8e4m3>);
static_assert(IsVectorOf<f8e5m2x4, f8e5m2>);
static_assert(IsVectorOf<f8e5m2x2, f8e5m2>);
static_assert(IsVectorOf<f32x4, f32>);
static_assert(IsVectorOf<f32x2, f32>);
static_assert(IsVectorOf<f16x2, f16>);
static_assert(IsVectorOf<bf16x2, bf16>);
static_assert(IsVectorOf<f64x2, f64>);
static_assert(IsVectorOf<i8x4, i8>);
static_assert(IsVectorOf<i8x2, i8>);
static_assert(IsVectorOf<u8x4, u8>);
static_assert(IsVectorOf<u8x2, u8>);
static_assert(IsVectorOf<i16x4, i16>);
static_assert(IsVectorOf<i16x2, i16>);
static_assert(IsVectorOf<u16x4, u16>);
static_assert(IsVectorOf<u16x2, u16>);
static_assert(IsVectorOf<i32x4, i32>);
static_assert(IsVectorOf<i32x2, i32>);
static_assert(IsVectorOf<u32x4, u32>);
static_assert(IsVectorOf<u32x2, u32>);
static_assert(IsVectorOf<i64x4, i64>);
static_assert(IsVectorOf<i64x2, i64>);
static_assert(IsVectorOf<u64x4, u64>);
static_assert(IsVectorOf<u64x2, u64>);
