# 当前 MA-CNN-A 三协议实验结果

最后核验：2026-08-13

## 1. 结果归属

本目录记录的是当前 532,166 参数 MA-CNN-A，而不是历史 486,838 参数 V1。
当前网络采用：

- 三个非对称卷积分支，尺度为 `8/16/32`；
- 每个分支通道依次为 `32 -> 32 -> 64 -> 64`；
- 六路 ECA 风格通道注意力；
- 注意力加权后使用 `1x8/8x1`、98 通道分类卷积；
- GAP 后接四分类线性层。

历史 V1 的片段级 Accuracy 为 95.15%，仅保存在
`results/legacy_paper_reproduction/` 中，不能再作为当前网络的精度。

## 2. 统一实验配置

- 输入：16 kHz、3 秒音频片段、`64x94` log-Mel；
- 数据量：每类 train/validation/test 为 `3500/1000/500`；
- 优化器：SGD，momentum 0.9；
- 学习率：`0.001` warmup 至 `0.01`，之后 cosine decay 至 `1e-5`；
- batch size：16；
- early stopping patience：10；
- 完整 model seed：42、43；
- 三种协议使用各自冻结的 split manifest，模型、特征和训练配置保持一致。

## 3. 正式结果

以下为 seed 42/43 的均值 +/- 样本标准差：

| 数据协议 | Segment Accuracy | Recording Accuracy | Vessel-group Accuracy |
|---|---:|---:|---:|
| `segment_level` | **97.30 +/- 0.21%** | 98.24 +/- 0.31% | 99.03 +/- 0.00% |
| `recording_disjoint` | **66.20 +/- 0.49%** | **70.19 +/- 4.08%** | 66.67 +/- 3.14% |
| `vessel_name_disjoint` | **50.98 +/- 0.18%** | 56.25 +/- 2.21% | **53.13 +/- 4.42%** |

跨协议最公平的单变量比较是 Segment Accuracy：

```text
97.30% (随机片段)
   -> 66.20% (原始录音隔离)
   -> 50.98% (规范船名组隔离)
```

对应的 Macro-F1 为：

| 数据协议 | Segment Macro-F1 | Recording Macro-F1 | Vessel-group Macro-F1 |
|---|---:|---:|---:|
| `segment_level` | 97.30 +/- 0.21% | 98.30 +/- 0.35% | 99.33 +/- 0.00% |
| `recording_disjoint` | 66.21 +/- 0.50% | 70.50 +/- 3.82% | 67.64 +/- 2.64% |
| `vessel_name_disjoint` | 49.94 +/- 1.09% | 53.74 +/- 0.09% | 50.49 +/- 2.09% |

## 4. 单次运行明细

| 协议 | Seed | Best epoch | Best val Accuracy | Segment Accuracy | 协议主聚合 Accuracy |
|---|---:|---:|---:|---:|---:|
| segment | 42 | 40 | 97.20% | 97.45% | recording 98.46% |
| segment | 43 | 50 | 97.33% | 97.15% | recording 98.02% |
| recording | 42 | 18 | 66.25% | 66.55% | recording 67.31% |
| recording | 43 | 26 | 65.45% | 65.85% | recording 73.08% |
| vessel-name | 42 | 17 | 56.15% | 50.85% | vessel group 56.25% |
| vessel-name | 43 | 6 | 53.38% | 51.10% | vessel group 50.00% |

六次运行均存在完整的 `run_complete.json`、split validation、checkpoint、预测文件和
segment/recording/vessel 三层指标。原始完整产物位于 Git 忽略的 `runs/` 目录。

`runs/logs/segment_level_seed44.log` 只训练到第 63 轮，没有测试指标和完成标记，故不纳入
正式统计。当前均值反映同一冻结 split 下的模型初始化/训练随机性，不包含不同 group split
带来的划分不确定性。

## 5. 结论边界

- `segment_level` 明确存在原始录音交叉，只能表示同分布片段识别能力。
- `recording_disjoint` 保证同一 WAV 不跨集合，但同名船只仍可能跨集合。
- `vessel_name_disjoint` 保证规范船名组不跨集合，但船名组未经完整 MMSI/IMO 核验，不能称为
  严格物理船只身份隔离。
- 当前项目对外应报告 97.30 +/- 0.21% 的随机片段结果，并同时说明录音隔离和船名组隔离下的
  性能分别下降至 66.20 +/- 0.49% 和 50.98 +/- 0.18%。
