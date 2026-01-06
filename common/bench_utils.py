from __future__ import annotations

import time
from typing import Callable

import torch


torch.set_grad_enabled(False)




def run_benchmark(
    perf_func: Callable[[], torch.Tensor],
    tag: str,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
) -> tuple[torch.Tensor, float]:
    if iters <= 0:
        raise ValueError("iters must be > 0")

    # Warmup
    for _ in range(warmup):
        _ = perf_func()

    torch.cuda.synchronize()

    start = time.perf_counter()
    res: torch.Tensor = perf_func()
    for _ in range(iters - 1):
        res = perf_func()

    torch.cuda.synchronize()
    end = time.perf_counter()

    total_ms = (end - start) * 1000.0
    mean_ms = total_ms / iters

    assert isinstance(res, torch.Tensor)
    out_info = f"out_{tag}"

    # NumPy doesn't support some Torch dtypes (e.g. bf16, float8). We only need a
    # tiny preview for sanity-checking, so avoid .numpy() and cast just the preview.
    preview = res.flatten().detach()[:2]
    if preview.is_floating_point() or preview.is_complex():
        preview = preview.to(torch.float32)
    out_val = preview.cpu().tolist()
    out_val = [round(float(v), 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_ms:.8f}ms")

    if show_all:
        print(res)

    return res, mean_ms
