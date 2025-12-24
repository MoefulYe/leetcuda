import time
from typing import Optional, Callable
import torch
import relu as lib  # type: ignore
torch.set_grad_enabled(False)


def run_benchmark(
    perf_func: Callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(x, out)
    else:
        for i in range(warmup):
            _ = perf_func(x)
    torch.cuda.synchronize()

    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(x, out)
    else:
        for i in range(iters):
            out = perf_func(x)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2] # type: ignore
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

for S, K in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    x = torch.randn((S, K)).cuda().float().contiguous()
    y = torch.zeros_like(x).cuda().float().contiguous()
    run_benchmark(lib.relu_f32, x, "f32", y)
    run_benchmark(lib.relu_f32x4, x, "f32x4", y)
    run_benchmark(torch.relu, x, "f32_th")

    print("-" * 85)
    x_f16 = x.half().contiguous()
    y_f16 = y.half().contiguous()
    run_benchmark(lib.relu_f16, x_f16, "f16", y_f16)
    run_benchmark(lib.relu_f16x2, x_f16, "f16x2", y_f16)
    run_benchmark(lib.relu_f16x8, x_f16, "f16x8", y_f16)
    run_benchmark(lib.relu_f16x8_pack, x_f16, "f16x8pack", y_f16)
    run_benchmark(torch.relu, x_f16, "f16_th")
    print("-" * 85)