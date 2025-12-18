# Week 04（Day 22–28）Softmax：Attention 的核心积木

## 本周目标

- [ ] 理解并能复述 softmax 的数值稳定与性能要点：`max-trick`、exp/sum、避免多次全量读写。
- [ ] 明确“softmax 为什么常常带宽瓶颈/规约瓶颈”。

## 任务清单

- [ ] 跑：`kernels/softmax`
- [ ] 固定 2–3 组 shape：至少覆盖一个较长序列维度（例如 `N=4096/8192` 类）
- [ ] 用 `ncu` 做一次 profile，关注 global memory 与规约相关指标

## 你要看懂的点

- [ ] `max -> exp -> sum -> normalize` 的稳定流程
- [ ] online/分块 softmax 的基本想法（即使代码里未完全实现，也要理解动机）

## 本周产出（必做）

- [ ] 一句话解释：“FlashAttention 为什么要做 block-wise softmax？”
- [ ] 一张小表：不同 shape 下 softmax 时间变化趋势（越长越慢的原因写清楚）
