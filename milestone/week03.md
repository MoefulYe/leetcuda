# Week 03（Day 15–21）规约 Reduce：同步与性能瓶颈

## 本周目标

- [x] 能写出/读懂常见 reduce 的并行结构（block 内规约 + 多阶段）。
- [ ] 初步用 `ncu` 验证瓶颈（带宽/同步/占用率/分歧）。

## 任务清单

- [x] 跑：`kernels/reduce`
  - [x] 读：`kernels/reduce/README.md`
  - [x] 跑：对应 `*.py` 测试脚本
- [ ] 至少用一次 `ncu` 对 reduce kernel 做 profile（先不追求所有指标）

## 你要看懂的点

- [x] shared memory 规约的基本结构
- [x] 线程同步点（`__syncthreads()`）的成本与必要性
- [-] bank conflict 的“可能性”（先知道概念即可，Week 8 会系统看）

## 本周产出（必做）

- [ ] 一次 `ncu` 输出的关键结论：选 1–3 个你看得懂的指标（例如 dram 吞吐、occupancy、stall reason）
- [x] 一条经验：reduce 中“增加并行度/减少同步/减少访存”的哪一个最影响你看到的结果
