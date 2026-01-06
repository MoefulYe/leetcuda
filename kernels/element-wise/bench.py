import torch

import element_wise as lib  # type: ignore

from common.bench_utils import run_benchmark

torch.set_grad_enabled(False)


Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

for S, K in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    a = torch.randn((S, K)).cuda().float().contiguous()
    b = torch.randn((S, K)).cuda().float().contiguous()
    c = torch.zeros_like(a).cuda().float().contiguous()

    def perf_f32() -> torch.Tensor:
        lib.elementwise_add_f32(a, b, c)
        return c

    def perf_f32x4() -> torch.Tensor:
        lib.elementwise_add_f32x4(a, b, c)
        return c

    run_benchmark(perf_f32, tag="f32")
    run_benchmark(perf_f32x4, tag="f32x4")
    run_benchmark(lambda: torch.add(a, b, out=c), tag="f32_th")

    print("-" * 85)
    a_f16 = a.half().contiguous()
    b_f16 = b.half().contiguous()
    c_f16 = c.half().contiguous()

    def perf_f16() -> torch.Tensor:
        lib.elementwise_add_f16(a_f16, b_f16, c_f16)
        return c_f16

    def perf_f16x2() -> torch.Tensor:
        lib.elementwise_add_f16x2(a_f16, b_f16, c_f16)
        return c_f16

    def perf_f16x8() -> torch.Tensor:
        lib.elementwise_add_f16x8(a_f16, b_f16, c_f16)
        return c_f16

    def perf_f16x8_pack() -> torch.Tensor:
        lib.elementwise_add_f16x8_pack(a_f16, b_f16, c_f16)
        return c_f16

    run_benchmark(perf_f16, tag="f16")
    run_benchmark(perf_f16x2, tag="f16x2")
    run_benchmark(perf_f16x8, tag="f16x8")
    run_benchmark(perf_f16x8_pack, tag="f16x8pack")
    run_benchmark(lambda: torch.add(a_f16, b_f16, out=c_f16), tag="f16_th")
    print("-" * 85)
