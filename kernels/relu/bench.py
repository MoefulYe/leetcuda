from typing import Any
import torch
import relu as lib  # type: ignore
lib: Any = lib

from bench_utils import run_benchmark

torch.set_grad_enabled(False)



Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

for S, K in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    x = torch.randn((S, K)).cuda().float().contiguous()
    y = torch.zeros_like(x).cuda().float().contiguous()

    def perf_f32() -> torch.Tensor:
        lib.relu_f32(x, y)
        return y

    def perf_f32x4() -> torch.Tensor:
        lib.relu_f32x4(x, y)
        return y

    run_benchmark(perf_f32, tag="f32")
    run_benchmark(perf_f32x4, tag="f32x4")
    run_benchmark(lambda: torch.relu(x), tag="f32_th")

    print("-" * 85)
    x_f16 = x.half().contiguous()
    y_f16 = y.half().contiguous()

    def perf_f16() -> torch.Tensor:
        lib.relu_f16(x_f16, y_f16)
        return y_f16

    def perf_f16x2() -> torch.Tensor:
        lib.relu_f16x2(x_f16, y_f16)
        return y_f16

    def perf_f16x8() -> torch.Tensor:
        lib.relu_f16x8(x_f16, y_f16)
        return y_f16

    def perf_f16x8_pack() -> torch.Tensor:
        lib.relu_f16x8_pack(x_f16, y_f16)
        return y_f16

    run_benchmark(perf_f16, tag="f16")
    run_benchmark(perf_f16x2, tag="f16x2")
    run_benchmark(perf_f16x8, tag="f16x8")
    run_benchmark(perf_f16x8_pack, tag="f16x8pack")
    run_benchmark(lambda: torch.relu(x_f16), tag="f16_th")
    print("-" * 85)