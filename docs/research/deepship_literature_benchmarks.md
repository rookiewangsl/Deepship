# DeepShip 三种隔离协议的文献基准

最后核验：2026-08-12

## 1. 结论先行

DeepShip 论文尚未形成统一的 segment、recording、vessel 三档官方 benchmark。不同论文即使都
报告 Accuracy，也可能使用不同切片长度、训练/测试比例、聚合层级和船只身份定义，不能只按
数字高低排序。

截至当前，适合本项目采用的参考框架为：

| 协议 | 文献报告 | 建议采用的比较口径 |
|---|---:|---|
| 随机切片、不隔离 | 约 97%～98.4% | 当前现代网络的常规高水平，但不代表新录音或新船只泛化 |
| 原始录音隔离 | 保守结果约 70%～75%；单篇最高报告 98.23% | 同时列出保守锚点和最高报告，不把 98.23% 当作稳定共识 |
| 船只隔离 | 尚无统一可核验 SOTA；严格化参考约 50%～65% | 明确该区间是综合现有更严格实验得到的参考，不是官方 benchmark |

## 2. 不隔离／随机切片

MA-CNN-A 论文将 DeepShip 切为 3 秒样本，构造每类 5000 个样本和
`3500/1000/500` 的 train/validation/test 集，报告 Accuracy **98.2%**。论文没有说明按原始
录音或船只分组，因此本项目将它归入 segment-level 常规协议。

2026 年的轻量多尺度门控专家网络在 DeepShip 上报告 **98.43±0.08%**；论文把 unseen-recording
和 MMSI-level vessel-disjoint 作为额外补充实验，因此摘要中的 98.43% 不能解释成严格隔离结果。

可靠表述是：现代 DeepShip 随机切片结果通常约为 **97%～98.4%**，但这一协议可能让同一原始
录音甚至同一船只的相关片段同时进入训练集和测试集。

## 3. 原始录音隔离

Hummel 等在 Applied Acoustics 2026 中将独立 DeepShip recordings 随机按 80:20 划分，并明确
保证同一录音不同时出现在训练集和测试集。其 Fig. 6 中最佳 DeepShip Accuracy 约 **73%**。

UATFSN 论文同样声明同一原始录音不跨训练/测试集，将录音切成 3 秒样本并报告
**98.23%**。论文称实验独立执行 10 次，但正文中的不同表格还出现 97.26%、98.19% 和 98.23%
等不同汇总值，因此该结果应当作为“已发表最高报告”单列，而不作为录音隔离的稳定行业水平。

录音隔离仍可能让同一艘物理船的不同 recordings 跨集合。DeepShip 包含 609 条录音和公开描述
中的 265 艘船，所以 recording-disjoint 不是 vessel-disjoint。

## 4. 船只隔离

Hummel 等采用 time-wise split：2017 年 11 月以前为训练集，2017 年 12 月以后为测试集；约
60% 测试样本来自训练阶段未见过的船只。该协议下监督 ResNet18 最好为 **63.07%**。后续
Ecological Informatics 2026 使用同类时间划分，BEATS linear probe 达到 **65.4%**。

这两项实验比随机录音划分更接近未知船只部署，但测试集中仍存在部分已见船只，因此不能称为
“所有测试船只均隔离”的严格 vessel-disjoint。

2026 年 7 月发表的轻量多尺度门控专家网络明确增加了 **MMSI-level vessel-disjoint** 实验，
这是目前方法学上最接近严格物理船只隔离的公开论文之一；但官方公开摘要没有给出该补充实验的
数值，当前不能据此填写可核验的严格船隔离 SOTA。

因此，**50%～65%** 只能作为本项目规划和结果解释时的合理参考区间。其中上沿来自带大量未见
船只的时间划分论文，下沿由更严格、全部名称组隔离时预期的进一步下降推断得到。论文中不能把
它写成已经建立的官方准确率区间。

## 5. 与本项目当前结果的对应

本项目使用同一 532,166 参数 V2 网络、同一特征和训练配置，只改变冻结 split manifest。当前
完成 seed 42/43：

| 协议 | 主要报告层级 | Accuracy（均值±样本标准差） | 与文献的关系 |
|---|---|---:|---|
| `segment_level` | segment | **97.30±0.21%** | 落在现代随机切片 97%～98.4% 范围内 |
| `recording_disjoint` | recording | **70.19±4.08%** | 接近 Hummel 随机录音划分约 73% 的保守锚点 |
| `vessel_name_disjoint` | vessel group | **53.13±4.42%** | 落在 50%～65% 的严格化参考范围内 |

论文直接横向比较时还应同时给出 segment Accuracy：三种协议分别为
**97.30±0.21%**、**66.20±0.49%** 和 **50.98±0.18%**。这是唯一完全相同评估层级的三协议
单变量对照。

`vessel_name_disjoint` 使用规范船名键，不是 MMSI/IMO。它可以表述为“未见规范船名组泛化”，
不能升级为已经验证的物理船只身份隔离。

## 6. 推荐写法

> 在相同网络、特征和训练设置下，随机切片测试准确率为 97.30±0.21%，录音隔离后片段准确率
> 降至 66.20±0.49%，船名组隔离后进一步降至 50.98±0.18%。录音级和船名组级聚合准确率
> 分别为 70.19±4.08% 和 53.13±4.42%。该趋势与近期文献中更严格划分导致显著性能下降的
> 观察一致。船名组协议未使用 MMSI/IMO，因此结论限定为规范船名组泛化。

## 7. 主要来源

1. [MA-CNN-A, Journal of Marine Science and Engineering, 2024](https://www.mdpi.com/2077-1312/12/1/130)
2. [DeepShip original SCAE paper, Expert Systems with Applications, 2021](https://www.sciencedirect.com/science/article/abs/pii/S0957417421007016)
3. [Generalized embeddings with contrastive learning, Applied Acoustics, 2026](https://ir.cwi.nl/pub/35751/35751.pdf)
4. [Linear probing with pretrained audio embeddings, Ecological Informatics, 2026](https://ir.cwi.nl/pub/36356/36356.pdf)
5. [UATFSN, Ocean Engineering, 2026](https://www.sciencedirect.com/science/article/pii/S0029801825029178)
6. [Lightweight multi-scale gated expert network, Ocean Engineering, 2026](https://www.sciencedirect.com/science/article/pii/S0029801826020603)
