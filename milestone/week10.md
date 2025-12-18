# Week 10（Day 64–70）FlashAttention：跑通 + Baseline 固化

## 本周目标

- [ ] 在 3080 上跑通 `kernels/flash-attn` 的一个 baseline（能跑、数值正确、可重复）。
- [ ] 固定 benchmark shape 与测量方法，为后续对比奠定基础。

## 任务清单

- [ ] 跑：`kernels/flash-attn` 提供的测试/benchmark（以目录 README 为准）
- [ ] 选 2–3 组固定 shape（建议包含 `N=2048/4096/8192`，`D=64`）
- [ ] 做一次 `nsys`（确认没有异常的同步/拷贝）
- [ ] 做一次 `ncu`（只跑最关键的 kernel，先抓大方向）

## 本周产出（必做）

- [ ] Baseline 表格：shape → 时间/TFLOPS（或相对速度）→ 一句结论
- [ ] 正确性说明：误差范围、对比对象（FA2/SDPA/自带 reference 任选其一）
