# Week 02（Day 8–14）向量化与访存：elementwise/relu/gelu

## 本周目标

- [ ] 对“向量化读写（float4/half2）为什么会更快”建立直觉。
- [ ] 能看懂并解释：对齐、越界处理、向量化带来的吞吐提升与限制。

## 任务清单

- [ ] 选择 2 个目录跑通（建议）
  - [ ] `kernels/relu`
  - [ ] `kernels/gelu`（或 `kernels/sigmoid`/`kernels/swish`）
- [ ] 对每个目录做一次小对比
  - [ ] naive vs vectorized（如果目录里提供）
  - [ ] 至少固定 2 组 shape（如 `S=4096,K=4096` 和 `S=1024,K=4096`）

## 你要看懂的点

- [ ] `half`/`half2`/`float4` 的 load/store（以及对齐要求）
- [ ] “pack/unpack”对 global memory 合并读写的影响
- [ ] kernel launch 参数（threads/blocks）如何影响 occupancy（先从现象认识）

## 本周产出（必做）

- [ ] 写一个对比表：每个 kernel 至少 2 种实现、2 组 shape、记录时间
- [ ] 准备 Week 3：你最想用 `ncu` 验证的一个问题（比如“为什么 pack 更快？”）
