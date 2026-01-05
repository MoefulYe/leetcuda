from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Optional, Tuple

import torch


torch.set_grad_enabled(False)


def _sync_if_cuda(tensors: Iterable[object]) -> None:
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            torch.cuda.synchronize()
            return


def _invoke_with_optional_out(
    perf_func: Callable[..., Any],
    args: Tuple[Any, ...],
    out: Optional[torch.Tensor],
) -> Any:
    if out is None:
        return perf_func(*args)

    # Prefer passing out positionally (matches existing custom extensions).
    try:
        return perf_func(*args, out)
    except TypeError:
        # Some callables already bind out via functools.partial (e.g., torch.add(out=...)).
        return perf_func(*args)


def run_benchmark(
    perf_func: Callable[..., Any],
    *args: Any,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
) -> tuple[torch.Tensor, float]:
    """Benchmark a CUDA op.

    - If `out` is provided, we try calling `perf_func(*args, out)`; otherwise `perf_func(*args)`.
    - If the callable returns None (common for out-parameter kernels), `out` is used as the result.

    Returns: (result_tensor, mean_time_ms)
    """

    if out is not None:
        out.zero_()

    # Warmup
    for _ in range(warmup):
        res = _invoke_with_optional_out(perf_func, args, out)
        if res is None:
            res = out

    _sync_if_cuda((out, *args))

    start = time.perf_counter()
    res: Any = None
    for _ in range(iters):
        res = _invoke_with_optional_out(perf_func, args, out)
        if res is None:
            res = out
    _sync_if_cuda((res, out, *args))
    end = time.perf_counter()

    total_ms = (end - start) * 1000.0
    mean_ms = total_ms / iters

    assert isinstance(res, torch.Tensor)
    out_info = f"out_{tag}"
    out_val = res.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(float(v), 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_ms:.8f}ms")

    if show_all:
        print(res)

    return res, mean_ms
