# Week 06（Day 36–42）LayerNorm/RMSNorm：规约 + 逐元素融合

## 本周目标

- [ ] 能复述 LN/RMSNorm 的计算图，知道哪里是规约、哪里是逐元素。
- [ ] 观察“融合前后”对访存的影响（哪怕只是理解层面）。

## 任务清单

- [ ] 跑：`kernels/layer-norm`
- [ ] 跑：`kernels/rms-norm`
- [ ] 固定 2 组 hidden size（例如 `D=1024/4096`），观察时间与瓶颈变化

## 你要看懂的点

- [ ] LN：mean/var 两个统计量，RMSNorm：只用平方和
- [ ] 规约策略对性能的影响：每次读写多少次、是否能复用缓存/寄存器

## 本周产出（必做）

- [ ] LN vs RMSNorm 的差异总结（计算/访存/潜在瓶颈）
- [ ] 标注 1 个你认为“未来可融合”的点（为 FlashAttention 的融合思路铺路）
