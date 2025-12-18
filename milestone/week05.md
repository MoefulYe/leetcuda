# Week 05（Day 29–35）Transpose/布局与带宽：为融合做铺垫

## 本周目标

- [ ] 建立“数据布局决定性能上限”的直觉。
- [ ] 能看懂 tile + shared memory 的 transpose 结构（不用写到极致优化）。

## 任务清单

- [ ] 跑：`kernels/mat-transpose`
- [ ] 选读：`kernels/swizzle`（理解布局/冲突概念即可）
- [ ] 用 `ncu` 对 transpose kernel 做一次 profile，关注读写合并、shared bank conflict（若能看到）

## 你要看懂的点

- [ ] naive transpose 为什么慢（不合并访问/写回散乱）
- [ ] tile transpose 为什么快（合并读写 + shared 中转）

## 本周产出（必做）

- [ ] 记录 1 次对比：naive vs tiled（或不同 tile 参数）
- [ ] 写下你观察到的 1 个“带宽相关”证据（例如 dram 吞吐接近上限、stall memory）
