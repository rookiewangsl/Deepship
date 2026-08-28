# DeepShip 隔离协议与基线文献

最后核验：2026-08-28

## 1. 结论先行

DeepShip 论文尚未形成统一的 segment、recording、vessel 三档官方 benchmark。不同论文即使都
报告 Accuracy，也可能使用不同切片长度、训练/测试比例、聚合层级和船只身份定义，不能只按
数字高低排序。

截至当前，适合本项目采用的参考框架为：

| 协议 | 保守／可复核锚点 | 已发表高结果 | 建议采用的比较口径 |
|---|---:|---:|---|
| 随机切片、不隔离 | 约 97%～98.2% | MA-CNN-A 98.2% | 用于验证原方法复现，不代表新录音或新船只泛化 |
| 原始录音隔离 | Transformer 类基线约 67%～70%；LHK-Net 72.27% | UATFSN 98.23%；MGAE-Net 98.43±0.08% | 同时列出保守锚点和高结果，不把接近 98% 当作稳定共识 |
| 时间隔离、多数测试船未见 | 无监督 Conformer 54.87%；BEATs 65.4% | BEATs 65.4% | 作为现实型未见船舶参照，但不能称为全部船只隔离 |
| MMSI 级船舶隔离 | 尚未形成跨论文统一区间 | MGAE-Net 95.83±0.07% | 作为严格 MMSI 协议下的已发表高性能上界，不与不同 manifest 直接排名 |

### 1.1 本项目推荐采用的核心基线

| 优先级 | 基线论文／模型 | DeepShip 协议 | 报告结果 | 对本项目的主要用途 | 结论边界 |
|---:|---|---|---:|---|---|
| 1 | MA-CNN-A（JMSE 2024） | 3 秒随机片段，未证明录音或船舶隔离 | 98.20% | 与当前 MA-CNN-A 的 `segment_level` 结果进行同结构复现比较 | 只能说明同分布片段识别能力 |
| 2 | LHK-Net 及其统一重训的 MobileViT-XS/UACTC（JMSE 2026） | 原始文件划分后再切 6 秒片段 | LHK-Net 72.27%；Transformer 类基线约 67%～70% | 作为 `recording_disjoint` 的主要保守外部基准 | LHK-Net 本身是 CNN；Transformer 数值由图中读取，按约数引用 |
| 3 | BEATs linear probe（Ecological Informatics 2026） | 按时间划分；测试集中多数船舶未在训练集出现 | 65.40% | 作为现实型未见船舶泛化锚点 | 测试集仍可能包含少量已见船舶，不是完全 vessel-disjoint |
| 4 | MGAE-Net（Ocean Engineering 2026） | 原始录音隔离；补充 MMSI 级船舶隔离 | 98.43±0.08%；MMSI 95.83±0.07% | 作为严格录音／MMSI 协议下的已发表高性能上界和后续结构改进参照 | 使用独立 split、双分辨率输入、HGRS 和复合损失，不能与当前结果直接排名 |

其中，MA-CNN-A 用于回答“是否复现原方法”，LHK-Net 用于回答“录音隔离结果是否落在可信区间”，
BEATs 用于回答“面对大量未见船舶时的现实性能如何”，MGAE-Net 用于表示“严格 MMSI 协议下的
最新高性能报告”。四者承担不同角色，不应合并成一个按 Accuracy 排序的榜单。

## 2. 不隔离／随机切片

MA-CNN-A 论文将 DeepShip 切为 3 秒样本，构造每类 5000 个样本和
`3500/1000/500` 的 train/validation/test 集，报告 Accuracy **98.2%**。论文没有说明按原始
录音或船只分组，因此本项目将它归入 segment-level 常规协议。

可靠表述是：现代 DeepShip 随机切片结果通常约为 **97%～98.2%**，但这一协议可能让同一原始
录音甚至同一船只的相关片段同时进入训练集和测试集。

## 3. 原始录音隔离

Hummel 等在 Applied Acoustics 2026 中将独立 DeepShip recordings 随机按 80:20 划分，并明确
保证同一录音不同时出现在训练集和测试集。其 Fig. 6 中最佳 DeepShip Accuracy 约 **73%**。

LHK-Net 论文先划分原始音频文件，再将其切为 6 秒片段，并在同一个 file-level split 下统一重训
MobileViT-XS 和 UACTC。其 Transformer 类基线约为 **67%～70%**，LHK-Net 为 **72.27%**。
这组结果是当前项目 `recording_disjoint` 最适合采用的保守外部锚点。

UATFSN 论文同样声明同一原始录音不跨训练/测试集，将录音切成 3 秒样本并报告
**98.23%**。论文称实验独立执行 10 次，但正文中的不同表格还出现 97.26%、98.19% 和 98.23%
等不同汇总值，因此该结果应当作为“已发表最高报告”单列，而不作为录音隔离的稳定行业水平。

MGAE-Net 使用 7:3 原始录音划分、3 秒非重叠片段和五个模型种子，报告
**98.43±0.08%**。它进一步采用双分辨率 log-Mel、困难－随机样本重构和样本级门控，因而应当
作为高性能上界单列，而不能仅凭 Accuracy 与本项目的冻结 manifest 结果直接排序。

录音隔离仍可能让同一艘物理船的不同 recordings 跨集合。DeepShip 包含 609 条录音和公开描述
中的 265 艘船，所以 recording-disjoint 不是 vessel-disjoint。

## 4. 船只隔离

Hummel 等采用 time-wise split：2017 年 11 月以前为训练集，2017 年 12 月以后为测试集；约
60% 测试样本来自训练阶段未见过的船只。该协议下监督 ResNet18 最好为 **63.07%**。后续
Ecological Informatics 2026 使用同类时间划分，BEATS linear probe 达到 **65.4%**。

这两项实验比随机录音划分更接近未知船只部署，但测试集中仍存在部分已见船只，因此不能称为
“所有测试船只均隔离”的严格 vessel-disjoint。

MGAE-Net 进一步按 MMSI 在每个类别内划分船只，保证同一 MMSI 的所有录音仅属于 train、validation
或 test 中的一个集合。其划分包含 157/28/80 个训练、验证、测试 MMSI，在该协议下 MGAE-Net 为
**95.83±0.07%**，统一重训的 UATR-Transformer 为 **94.02±0.07%**。这是目前方法学上最接近
严格物理船只隔离的公开实验之一。

需要同时保留两类参照：Hummel/BEATs 的 **54.87%～65.4%** 是现实型时间隔离下的保守锚点；
MGAE-Net 的 **95.83%** 是独立 MMSI split、输入和训练管线下的高性能报告。两者差异很大，说明
DeepShip 的 vessel-level 结果仍未形成统一 benchmark，不能把某一个数值写成已经建立的 SOTA 区间。

## 5. 与本项目当前结果的对应

本项目使用同一 532,166 参数 V2 网络、同一特征和训练配置，只改变冻结 split manifest。当前
完成 seed 42/43：

| 协议 | Segment Accuracy | 协议主聚合 Accuracy | 最合适的文献参照 | 比较结论 |
|---|---:|---:|---|---|
| `segment_level` | **97.30±0.21%** | segment 97.30±0.21% | MA-CNN-A 98.20% | 基本复现原方法；不能代表泛化 |
| `recording_disjoint` | **66.20±0.49%** | recording 70.19±4.08% | Transformer 类约 67%～70%；LHK-Net 72.27% | 片段指标接近严格重训的 Transformer 基线，低于 LHK-Net 约 6 个百分点 |
| `vessel_name_disjoint` | **50.98±0.18%** | vessel group 53.13±4.42% | BEATs 65.4%；MGAE-Net MMSI 95.83±0.07% | 前者为现实型保守锚点，后者为不同 MMSI split 下的高性能上界 |

论文直接横向比较时应优先使用上表中的 Segment Accuracy，因为这是三种协议间唯一完全相同的
评估层级。Recording 和 vessel-group 聚合结果应作为部署层面的补充指标，不能与文献中的 segment
Accuracy 混在同一列直接排序。

`vessel_name_disjoint` 使用规范船名键，不是 MMSI/IMO。它可以表述为“未见规范船名组泛化”，
不能升级为已经验证的物理船只身份隔离。

## 6. 推荐写法

> 在相同网络、特征和训练设置下，随机切片测试准确率为 97.30±0.21%，录音隔离后片段准确率
> 降至 66.20±0.49%，船名组隔离后进一步降至 50.98±0.18%。录音级和船名组级聚合准确率
> 分别为 70.19±4.08% 和 53.13±4.42%。该趋势与近期文献中更严格划分导致显著性能下降的
> 观察一致。船名组协议未使用 MMSI/IMO，因此结论限定为规范船名组泛化。

文献对比段建议补充：

> 在原始录音隔离条件下，当前 MA-CNN-A 的片段准确率为 66.20±0.49%，与统一 file-level 协议下
> 约 67%～70% 的 Transformer 类基线接近，并低于 LHK-Net 的 72.27%。在未见船舶相关评测中，
> BEATs 在时间隔离且测试集多数船舶未见的条件下达到 65.4%；MGAE-Net 在其独立 MMSI-level
> split 下报告 95.83±0.07%。由于船只身份定义、split manifest、输入表示和训练管线均不同，后两项
> 分别作为保守锚点和高性能上界，而不与本项目的船名组隔离结果进行直接排名。

## 7. 主要来源

1. [MA-CNN-A, Journal of Marine Science and Engineering, 2024](https://www.mdpi.com/2077-1312/12/1/130)
2. [DeepShip original SCAE paper, Expert Systems with Applications, 2021](https://www.sciencedirect.com/science/article/abs/pii/S0957417421007016)
3. [Generalized embeddings with contrastive learning, Applied Acoustics, 2026](https://ir.cwi.nl/pub/35751/35751.pdf)
4. [Linear probing with pretrained audio embeddings, Ecological Informatics, 2026](https://ir.cwi.nl/pub/36356/36356.pdf)
5. [UATFSN, Ocean Engineering, 2026](https://www.sciencedirect.com/science/article/pii/S0029801825029178)
6. [Lightweight multi-scale gated expert network, Ocean Engineering, 2026](https://www.sciencedirect.com/science/article/pii/S0029801826020603)
7. [LHK-Net, Journal of Marine Science and Engineering, 2026](https://www.mdpi.com/2077-1312/14/17/1561)
