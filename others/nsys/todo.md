# CUDA Profiling 学习 TODO（Nsight Systems + Nsight Compute）

> 目标：把 profiling 变成可复现的“选瓶颈→定位→验证→对比”的工作流。
> 建议节奏：先 nsys（应用级）再 ncu（内核级），最后再做自动化。

## 阶段 0：环境与基线（2–4 小时）

- [ ] 确认工具可用：`nsys --version`、`ncu --version`
- [ ] 用 release 配置编译（避免 `-G`），确保符号/行号信息可用于源码关联
- [ ] 选 1 个基线程序（建议从本仓库 `kernels/*/bench.py` 里挑一个）
- [ ] 跑通一次全流程：
  - [ ] 生成 1 份 Nsight Systems 报告（`.nsys-rep`）
  - [ ] 生成 1 份 Nsight Compute 报告（`.ncu-rep`）
- [ ] 基线记录（写在这里也行）：
  - [ ] 机器信息：GPU 型号 / 驱动 / CUDA 版本
  - [ ] 程序输入规模（N、batch、iters 等）
  - [ ] 基线耗时（端到端 + 关键 kernel）

## 阶段 1：Nsight Systems（应用/系统级，大图定位）（4–8 小时）

- [ ] 学会从时间线回答这些问题：
  - [ ] 时间主要花在 CPU、CUDA API、Memcpy、Kernel 还是同步？
  - [ ] GPU 是否存在明显空转区间？空转由什么引起（同步/依赖/线程调度）？
  - [ ] 是否有可并行的机会（多 stream / 重叠拷贝计算 / 减少同步）？
- [ ] 对以下任意 2 个基准各跑一份 nsys：
  - [ ] `kernels/element-wise/bench.py`
  - [ ] `kernels/relu/bench.py`
  - [ ] `kernels/reduce/bench.py`
- [ ] 产出：
  - [ ] 列出 Top 1–3 的“最值得优化对象”（kernel 或 memcpy 或同步点）
  - [ ] 给出每个对象的“证据链一句话”（从时间线/统计得出）

## 阶段 2：Nsight Compute（内核级，指标→结论→改动→复测）（10–18 小时）

- [ ] 指标方法论（每次分析都按这个顺序写结论）：
  - [ ] 先判断 compute-bound vs memory-bound
  - [ ] 再定位：访存效率/吞吐、占用率、分支、指令效率、L2/Cache 行为
  - [ ] 最后给出 1 个可验证的优化假设
- [ ] 跟做材料（从仓库开始）：
  - [ ] `others/nsight-training/cuda/nsight_compute/vlog_memory_workload/`
  - [ ] `others/nsight-training/cuda/2020_ncu_smem/`（`t5_*.cu`）
- [ ] 选择 1 个 kernel 做完整闭环（建议顺序由易到难）：
  - [ ] element-wise
  - [ ] relu
  - [ ] reduce
- [ ] 每个闭环必须产出：
  - [ ] Baseline 报告（ncu）
  - [ ] 1 次改动后的报告（ncu）
  - [ ] 对比结论：哪些指标变化支撑“真的变好/变坏”
  - [ ] 性能对比：至少记录 kernel 时间变化（最好也记端到端变化）

## 阶段 3：把 profiling 变成工作流（4–8 小时，可选但很值）

- [ ] 报告与回归：
  - [ ] 给关键 kernel 固定输入规模与运行参数
  - [ ] 把 profiling 命令封装为脚本（可重复跑）
- [ ] 自动化分析（可选）：
  - [ ] 参考 `others/nsight-training/cuda/nsight_compute/python_report_interface/`
  - [ ] 写一个小脚本：自动提取 3–5 个关键指标并输出表格

## 完成标准（你可以把它当“毕业条件”）

- [ ] 能用 nsys 快速选出值得优化的对象（Top-N）
- [ ] 能用 ncu 对一个 kernel 写出“证据链”并完成至少 1 次有效优化
- [ ] 能复现/对比：同一输入规模下，优化前后报告与性能变化一致
