# Milestones (100天 / 按周学习计划)

面向：RTX 3080（Ampere, `sm_86`）+ NixOS + `nix flake` + `uv venv` 的学习路径。目标是在 100 天内完成：

- [ ] CUDA 基础（线程模型 / 内存访问 / 常见算子）
- [ ] 周边工具链（`nsys`/`ncu`）能独立定位瓶颈
- [ ] FlashAttention 主线学完整，并做 1 次“有效的小优化闭环”（提出假设 → 改动 → profile 证明 → 解释结果）

## 使用方式

- 每周对应一个文件：`milestone/weekXX.md`
- 每周建议投入：每天约 2 小时（6 天推进 + 1 天复盘/补漏）

## 通用约定（每周都做）

- [ ] 固定架构（3080）：`export TORCH_CUDA_ARCH_LIST="8.6"`
- [ ] 每个主题按同一 workflow：读 `kernels/<name>/README.md` → 跑 `python3 <name>.py` → 用 `ncu/nsys` 做一次 profile → 记录结论。
- [ ] 每周产出（强烈建议）：1 份本周笔记（你可写到自己的笔记系统，不一定提交到仓库）：
  - [ ] 基准 shape（输入规模）
  - [ ] 时间、吞吐（TFLOPS/GB/s 任选其一即可）
  - [ ] `ncu`/`nsys` 看到的 top bottleneck（1–3 条）
  - [ ] 你下周要验证的 1 个问题

## 周计划索引

- `milestone/week01.md`：环境与第一条闭环（elementwise）
- `milestone/week02.md`：向量化与访存（relu/gelu 等）
- `milestone/week03.md`：规约（reduce）
- `milestone/week04.md`：Softmax（为 Attention 铺路）
- `milestone/week05.md`：Transpose/布局与带宽
- `milestone/week06.md`：LayerNorm/RMSNorm
- `milestone/week07.md`：RoPE/Embedding（Attention 周边积木）
- `milestone/week08.md`：Nsight 工具链专项（nsys+ncu）
- `milestone/week09.md`：FlashAttention：概念与代码导读
- `milestone/week10.md`：FlashAttention：跑通 + baseline 固化
- `milestone/week11.md`：FlashAttention：策略对比与读写路径
- `milestone/week12.md`：FlashAttention：深入 profile，定位主要瓶颈
- `milestone/week13.md`：FlashAttention：一次小优化实验（闭环）
- `milestone/week14.md`：迁移到实验室 GPU + 总结
- `milestone/week15.md`：Day 99–100：收尾清单
