# Reduce (Sum) Benchmark Report

## Summary
- Bench compares multiple custom CUDA reduce implementations against PyTorch `torch.sum` baseline.
- For FP16/BF16 inputs, **accumulating into FP32** (`*_f32`) matches `torch.sum(..., dtype=torch.float32)` much better than accumulating into FP16/BF16 (`*_f16`, `*_bf16`).
- The variants that use **vector/pack loads + FP32 accumulation** (e.g. `x2_f32`, `x8_f32`) are typically the fastest among custom kernels for the tested shapes.

## What is being benchmarked
- Input shapes: a set of 2D tensors `(S, K)`.
- Output: a single-element tensor `{1}` containing the sum of all elements.
- Implementations (from the `reduce` extension):
  - FP32: `reduce_scalar_sum_f32_f32`, `reduce_vector_sum_f32x4_f32`
  - FP16: scalar/vector/pack, with accumulation in FP16 or FP32 (`*_f16` vs `*_f32`)
  - BF16: scalar/vector/pack, with accumulation in BF16 or FP32 (`*_bf16` vs `*_f32`)
- PyTorch baselines:
  - Same-dtype accumulation/output: `torch.sum(x)`
  - FP32 accumulation/output: `torch.sum(x, dtype=torch.float32)`

## Key Observations
### 1) Low-precision accumulation is numerically different
- Kernels that accumulate into FP16/BF16 (`*_f16`, `*_bf16`) use low-precision arithmetic for the final sum.
- Because the implementation accumulates partial sums across blocks into a **single scalar** with `atomicAdd`, the rounding error can compound, so results can deviate noticeably from FP32 baselines.
- This is expected and aligns with the FP32-accumulation variants (`*_f32`) and `torch.sum(..., dtype=torch.float32)` being much closer.

### 2) Low-precision atomic accumulation is also slower
- The design accumulates block-level partial sums into a single output via `atomicAdd`.
- When `S` is large, many blocks contend on one address (high atomic contention).
- Atomic operations in FP16/BF16 paths can be significantly slower and under heavy contention the slowdown is amplified.
- Result: `*_bf16` / `*_f16` variants may be much slower than `*_f32` variants and even slower than PyTorch baselines for larger shapes.

### 3) Pack/vector + FP32 accumulation tends to be best
- `x2_f32` / `x8_f32` reduce the number of loads/instructions and keep accumulation in FP32.
- With the last step still being a single `atomicAdd` per block, these versions often strike the best balance between memory efficiency and atomic overhead.

## Notes on “fair” baselines
- If the goal is **numerical correctness** (relative to a stable reference), compare against `torch.sum(x, dtype=torch.float32)` and/or cast-to-fp32 outputs.
- If the goal is **final dtype behavior** (FP16/BF16 output), compare against `torch.sum(x)` (which returns the same dtype by default).

## Next Steps / Potential Improvements
- To improve both speed and accuracy for FP16/BF16 accumulation, avoid global scalar atomic contention:
  - Two-stage reduction: write per-block partial sums to an intermediate buffer, then run a second kernel to reduce that buffer.
  - Or use established reduce primitives (e.g., CUB-style block reduction patterns) under the same interface.
