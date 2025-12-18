# Week 09（Day 57–63）FlashAttention：概念与代码导读

## 本周目标

- [ ] 理解 FlashAttention 的核心动机：减少 HBM 往返、避免显式 materialize `QK^T`、在线更新 softmax。
- [ ] 读懂 `kernels/flash-attn` 中 1 条实现路线的大体结构（不追细节性能）。

## 任务清单

- [ ] 读：`kernels/flash-attn/README.md`
- [ ] 浏览代码：确定以下内容在代码里对应什么
  - [ ] tiling 参数（Br/Bc/D 等）
  - [ ] Q/K/V 的加载与缓存位置（global/shared/register）
  - [ ] softmax 的在线更新
- [ ] 先不改代码，只要能“讲清楚它在干什么”

## 本周产出（必做）

- [ ] 一张“FA forward 数据流图”（文本/手绘均可）：Q/K/V 如何进入、如何计算、如何写回 O
- [ ] 列出 3 个你觉得可能影响性能的 knob（后面会用来做实验）
