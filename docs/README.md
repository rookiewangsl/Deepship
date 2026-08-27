# DeepShip 文档导航

这里集中维护项目的研究路线、实验协议、运行说明和背景调研。文档按用途分类；训练代码、冻结
manifest 和实验产物仍分别位于 `src/`、`scripts/`、`protocols/`、`results/` 与 `runs/`。

## 当前主线

1. 从 [Conformer 实验与外测计划](experiments/conformer_experiment_plan.md) 了解研究问题、模型路线、
   消融矩阵、决策门和可能形成的结论。
2. 在 Linux/RTX 4070 上执行前，使用
   [Wav2Vec2-Conformer 基线运行说明](guides/conformer_baseline_linux.md) 核对环境和命令。
3. CNN 的三种隔离协议、既有结果和公平比较边界见
   [录音级与船名组级隔离训练路线](experiments/strict_isolation_training_plan.md)。

## 实验设计 `experiments/`

- [Conformer 实验、数据与外部评测计划](experiments/conformer_experiment_plan.md)：后续项目主路线，
  包含通用预训练、领域自监督适配、监督微调、消融和 PORTIA 外测。
- [录音级与船只级隔离训练路线](experiments/strict_isolation_training_plan.md)：CNN 三协议设计、冻结
  manifest、已完成结果与结论边界。

## 运行指南 `guides/`

- [Linux/RTX 4070 Conformer 基线](guides/conformer_baseline_linux.md)：服务器目录、依赖、预检、
  smoke test 和正式训练入口。
- [Windows GPU 三协议训练指南](guides/windows_training_guide.md)：既有 MA-CNN-A 隔离实验复现说明。
- [存储布局](guides/storage_layout.md)：代码、原始数据、缓存、checkpoint 和运行产物的放置原则。

## 调研与分析 `research/`

- [DeepShip 文献基准](research/deepship_literature_benchmarks.md)：不同隔离协议下结果的可比边界。
- [拖曳式 DAS 分布式光缆水听器摘要](research/das_towed_distributed_hydrophone_summary.md)：传感原理、
  域偏移和接入 DeepShip 路线。

## 图表与归档

- `figures/`：文档和论文架构图。
- `archive/`：不代表当前方案的本地历史备忘；默认不纳入 Git 和主导航。

文档中的性能数字必须同时说明数据划分、聚合层级和 seed 数量。尚未训练的模型只能表述为工程
验证完成，不能写成已经获得性能提升。
