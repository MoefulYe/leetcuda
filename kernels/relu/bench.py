import os
import sys
import torch
import relu as lib  # type: ignore

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from bench_utils import run_benchmark  # noqa: E402
torch.set_grad_enabled(False)



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