# Week 01（Day 1–7）环境与第一条闭环

## 本周目标

- [ ] 在 NixOS 上跑通本仓库至少 1 个 kernel 的完整闭环：编译 → 运行 → 正确性对齐 → 计时。
- [ ] 建立“固定架构编译”的习惯，避免编译全架构耗时过长。

## 任务清单

- [x] 准备开发环境（`nix develop` + `uv venv`）
- [x] 验证工具可用：`python`、`torch.cuda.is_available()`、`nvcc --version`
- [x] 固定编译架构：`export TORCH_CUDA_ARCH_LIST="8.6"`
- [ ] 跑通 `kernels/elementwise`
  - [ ] 读：`kernels/elementwise/README.md`
  - [ ] 跑：`python3 kernels/elementwise/elementwise.py`（或进入目录执行）

## 你要看懂的点（够用即可）

- [ ] grid/block/thread 的映射方式
- [ ] 一维 index 访问模式是否 coalesced（同一 warp 访问是否连续）
- [ ] Python bindings 大致做了什么（加载 extension、调用 kernel、对齐 PyTorch 输出）

## 本周产出（必做）

- [ ] 一次运行日志：记录 1–2 组 shape 的时间对比（自实现 vs PyTorch）
- [ ] 一句话结论：瓶颈更像“算力”还是“带宽”（先凭直觉即可，下周用 `ncu` 验证）
