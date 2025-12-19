# Element-wise Add Bench Report

## Summary
- Benchmark covers FP32/FP16 variants (scalar and vectorized) and PyTorch baseline on several S×K shapes.
- Outputs match across implementations; FP16 shows expected quantization relative to FP32.
- FP16 vectorized kernels (`x2/x8/pack`) are faster than scalar FP16; `pack` is marginally best. FP32 vectorization shows little benefit.
- Performance gap shrinks as problem size grows; large shapes are bandwidth-bound, so implementations converge.

## Observations
- Small shapes (e.g., 1024×1024) benefit more from vectorization: fewer instructions and reduced launch/control overhead lead to noticeable speedups over scalar FP16.
- For larger matrices, all implementations reach similar DRAM throughput; memory bandwidth dominates, so vectorization only yields slight gains.
- PyTorch half kernel (`out_f16_th`) already uses half2-style optimization, explaining its parity with custom vectorized versions.
- `elementwise_add_f16x8_kernel` lacks read-side bounds checks; benchmarks use powers-of-two sizes, so no issue, but non-multiple-of-8 lengths would need padding or a tail path (as in `f16x8_pack`).

## Suggested Profiling (to validate reasoning)
- Nsight Compute: compare `f16` vs `f16x2/8` for small vs large shapes; inspect `dram__throughput`, `sm__inst_executed`, `warp_issue_stalled_*`, `achieved_occupancy` to confirm bandwidth vs instruction/launch bottlenecks.
- Nsight Systems: timeline to check kernel launch overhead share on small shapes and convergence on large shapes; ensure no hidden memcpy dominates.
