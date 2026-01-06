from typing import Any

import torch
import reduce as lib  # type: ignore

lib: Any = lib

from bench_utils import run_benchmark


torch.set_grad_enabled(False)


def _torch_sum(x: torch.Tensor, *, out_dtype: torch.dtype | None = None, accum_dtype: torch.dtype | None = None) -> torch.Tensor:
    y = torch.sum(x, dtype=accum_dtype)
    if out_dtype is not None:
        y = y.to(out_dtype)
    return y.reshape(1)


def _run_one_shape(S: int, K: int) -> None:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")

    x_f32 = torch.randn((S, K), device="cuda", dtype=torch.float32).contiguous()

    # f32
    run_benchmark(lambda: lib.reduce_scalar_sum_f32_f32(x_f32), tag="f32_f32")
    run_benchmark(lambda: lib.reduce_vector_sum_f32x4_f32(x_f32), tag="f32x4_f32")
    run_benchmark(lambda: _torch_sum(x_f32, out_dtype=torch.float32, accum_dtype=torch.float32), tag="f32_th")

    print("-" * 85)

    # f16
    x_f16 = x_f32.to(torch.float16).contiguous()
    run_benchmark(lambda: lib.reduce_scalar_sum_f16_f16(x_f16), tag="f16_f16")
    run_benchmark(lambda: lib.reduce_scalar_sum_f16_f32(x_f16), tag="f16_f32")
    run_benchmark(lambda: lib.reduce_vector_sum_f16x2_f16(x_f16), tag="f16x2_f16")
    run_benchmark(lambda: lib.reduce_vector_sum_f16x2_f32(x_f16), tag="f16x2_f32")
    run_benchmark(lambda: lib.reduce_pack_sum_f16x8_f16(x_f16), tag="f16x8_f16")
    run_benchmark(lambda: lib.reduce_pack_sum_f16x8_f32(x_f16), tag="f16x8_f32")
    run_benchmark(lambda: _torch_sum(x_f16, out_dtype=torch.float16, accum_dtype=None), tag="f16_th")
    run_benchmark(lambda: _torch_sum(x_f16, out_dtype=torch.float32, accum_dtype=torch.float32), tag="f16_th_f32")

    print("-" * 85)

    # bf16 (optional: only if supported on current GPU + PyTorch build)
    if hasattr(torch, "bfloat16"):
        try:
            x_bf16 = x_f32.to(torch.bfloat16).contiguous()
            run_benchmark(lambda: lib.reduce_scalar_sum_bf16_bf16(x_bf16), tag="bf16_bf16")
            run_benchmark(lambda: lib.reduce_scalar_sum_bf16_f32(x_bf16), tag="bf16_f32")
            run_benchmark(lambda: lib.reduce_vector_sum_bf16x2_bf16(x_bf16), tag="bf16x2_bf16")
            run_benchmark(lambda: lib.reduce_vector_sum_bf16x2_f32(x_bf16), tag="bf16x2_f32")
            run_benchmark(lambda: lib.reduce_pack_sum_bf16x8_bf16(x_bf16), tag="bf16x8_bf16")
            run_benchmark(lambda: lib.reduce_pack_sum_bf16x8_f32(x_bf16), tag="bf16x8_f32")
            run_benchmark(lambda: _torch_sum(x_bf16, out_dtype=torch.bfloat16, accum_dtype=None), tag="bf16_th")
            run_benchmark(lambda: _torch_sum(x_bf16, out_dtype=torch.float32, accum_dtype=torch.float32), tag="bf16_th_f32")
        except Exception as e:
            print(f"(skip bf16: {e})")

    print("-" * 85)


def main() -> None:
    Ss = [1024, 2048, 4096]
    Ks = [256, 512, 1024]
    for S in Ss:
        for K in Ks:
            _run_one_shape(S, K)


if __name__ == "__main__":
    main()
