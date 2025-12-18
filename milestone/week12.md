# Week 12（Day 78–84）FlashAttention：深入 profile，锁定主要瓶颈

## 本周目标

- [ ] 把“我觉得慢”变成“证据表明慢在 X”：带宽、占用率、bank conflict、寄存器压力、指令吞吐等。
- [ ] 为 Week 13 的小优化实验选择一个可控变量（只选一个）。

## 任务清单

- [ ] 对选定路线 + 固定 shape 做：
  - [ ] `nsys`：看 timeline（launch、sync、kernel 之间是否有空洞）
  - [ ] `ncu`：抓核心指标（dram throughput / occupancy / stall reasons）
- [ ] 记录 1 个最主要的瓶颈信号，并提出 1 个对应假设

## 本周产出（必做）

- [ ] 一页“瓶颈诊断”：你看到了什么指标 → 你推断的原因 → 你准备改什么
- [ ] Week 13 实验设计：只改一个 knob（例如 Br/Bc、stage、shared padding、向量化粒度）
