# Week 08（Day 50–56）Nsight 工具链专项：nsys + ncu

## 本周目标

- [ ] 形成稳定的 profile 流程：先 `nsys` 看系统层 timeline，再 `ncu` 看单 kernel 微观指标。
- [ ] 学会“只看少数关键指标就能定位问题”，避免淹没在报告里。

## 任务清单

- [ ] 跟着：`kernels/nvidia-nsight/README.md` 完整做一遍
- [ ] 选 2 个你已跑过的 kernel 目录（如 `relu` + `softmax`），分别做：
  - [ ] `nsys profile ...`
  - [ ] `ncu ...`

## 建议关注的指标（先掌握这几个）

- [ ] `nsys`：kernel launch 密度、CPU/GPU 同步点、memcpy 是否频繁
- [ ] `ncu`：dram throughput、achieved occupancy、top stall reasons、shared memory bank conflict（如可见）

## 本周产出（必做）

- [ ] 写一份“你的 profile SOP”（5–10 行即可）：你每次优化如何跑、看哪些指标、如何下结论
- [ ] 确认 NixOS 权限：如果 `ncu` 报 perf counter 权限问题，记录报错并准备修复（必要时再处理系统配置）
