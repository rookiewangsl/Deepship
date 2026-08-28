# MA-CNN-A 船名 embedding 诊断

最后核验：2026-08-28

状态：seed 42/43 已完成；仅使用冻结 `vessel_name_disjoint` 的 train/validation，未读取 test。

## 1. 目的

本实验检查当前 MA-CNN-A 的 98 维池化表示是否仍保留可跨录音识别的规范船名信息。它作为严格
隔离下性能下降的失败分析证据，用于解释模型是否学习了船名相关捷径；本项目不据此进入
JASA/MBAT 或其他 vessel-invariant 模块的实现。

## 2. 输入与执行

- 模型：当前 532,166 参数 MA-CNN-A；seed 42 最佳 epoch 17，seed 43 最佳 epoch 6；
- 协议：冻结 `vessel_name_disjoint` manifest，SHA-256
  `67aea271aa227acb58b45a96868207ac5fabebfe5ae63fe3ee59bf6be28c803d`；
- train：14,000 个片段、399 条录音、162 个船名组；
- validation：4,000 个片段、118 条录音、50 个船名组；
- 特征：通过 forward hook 读取 `AdaptiveAvgPool2d` 后的 98 维 embedding，不修改网络；
- 服务器数据：Samsung T7 的 `/mnt/t7/ProjectData/Deepship/datasets/DeepShip`；
- 资源：CPU、`nice=15`、2 个数学线程、DataLoader 0 worker，不使用 GPU；
- 双 seed 的 Silhouette 抽样、最近邻子集和 probe 留出录音统一使用诊断 seed 42。

## 3. 诊断定义

1. **类别 Silhouette**：检查 embedding 是否形成清晰类别簇；
2. **类内船名 Silhouette**：在每个类别内部计算船名簇，再按样本数加权；
3. **录音最近邻**：先将同一录音的片段 embedding 平均，再检查最近邻是否来自同一船名；
4. **跨录音船名线性探测**：只选择至少有两条录音的船名组；每组留出一条完整录音作为
   probe test，其余录音用于训练标准化逻辑回归。该 probe 只用于测量冻结 embedding 中可解码的
   船名信息。

## 4. 结果

| Split／指标 | Seed 42 | Seed 43 | 两 seed 均值 | 随机参照 |
|---|---:|---:|---:|---:|
| Train segment 类别 Silhouette | 0.620 | 0.437 | 0.529 | — |
| Train recording 类别 Silhouette | 0.726 | 0.484 | 0.605 | — |
| Train recording 同船最近邻率 | 23.31% | 22.56% | 22.93% | 8.58% |
| Train 跨录音船名 probe Accuracy（55 类） | 30.28% | 38.05% | 34.17% | 1.82% |
| Train probe balanced Accuracy | 26.51% | 26.79% | 26.65% | 1.82% |
| Validation segment 类别 Silhouette | 0.050 | 0.014 | 0.032 | — |
| Validation recording 类别 Silhouette | 0.001 | -0.075 | -0.037 | — |
| Validation recording 类内船名 Silhouette | 0.109 | 0.155 | 0.132 | — |
| Validation recording 同船最近邻率 | 26.27% | 33.90% | 30.08% | 19.89% |
| Validation 跨录音船名 probe Accuracy（14 类） | 48.03% | 47.05% | 47.54% | 7.14% |
| Validation probe balanced Accuracy | 39.34% | 36.23% | 37.79% | 7.14% |

Train 类内船名 Silhouette 为负（seed 42/43 分别为 -0.308/-0.326），说明大量、非均衡的船名组
并未形成统一的凸形独立簇。因此不能只根据 Silhouette 声称“模型完全按船名聚类”。但是，录音
最近邻同船率稳定高于随机参照，跨录音线性 probe 也大幅高于机会水平，说明同一个 98 维表示中
确实包含可跨录音解码的船名特征。

Validation 的类别 Silhouette 接近零或为负，而同船最近邻和船名 probe 仍高于随机水平。这表明
模型面对训练阶段未见的船名组时，类别边界明显退化，但新的船舶个体信息仍能在表示中形成可预测
结构。该现象与 JASA 的“个体船舶特征干扰类别泛化”假设方向一致。

## 5. 结论与项目用途

当前证据表明 CNN embedding 中存在可跨录音解码的船名信息，支持“模型可能利用个体船舶相关
捷径”的解释，但不能证明该信息是严格隔离性能下降的唯一原因，也不能证明身份不变训练必然有效。

本项目不继续实现 JASA-lite、GRL、船舶对判别器或 momentum queue。该诊断保留用于：

1. 解释随机片段、未见录音和未见船名协议之间的性能落差；
2. 对比最终 CNN 与 Wav2Vec2-Conformer 的表示和错误模式；
3. 在项目报告中展示从指标异常到表征诊断、再到停止高成本结构改造的决策过程。

完整原始报告保存在：

- `runs/cnn_embedding_diagnostic/seed42/diagnostic_report.json`
- `runs/cnn_embedding_diagnostic/seed43/diagnostic_report.json`
