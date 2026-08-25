# DeepShip Conformer 实验、数据与外部评测计划

最后更新：2026-08-26
状态：实验方案已整理；B3 工程基线已实现但尚未下载权重、执行前向或训练，不代表已有实验结果

## 0. 当前实施状态

第一版可运行基线已经落地，Linux/RTX 4070 的环境准备、预检、smoke 和正式运行顺序见
[`conformer_baseline_linux.md`](conformer_baseline_linux.md)。本次只完成代码、配置与静态验证，
没有在当前 Mac 上安装深度学习依赖、下载约 2.5 GB 预训练权重或启动网络训练。

已实现范围：

- 使用原有冻结隔离 manifest，不在训练时重新划分数据；
- 以 3 s anchor 中心扩展 3/10/20/30 s 原始波形上下文；
- 官方预训练 Wav2Vec2-Conformer large、ASP 四分类头和分阶段解冻；
- checkpoint/resume、segment/recording/vessel-group 三级评测和可复现实验记录；
- 无下载、无训练的服务器环境与数据预检。

尚未实现范围仍包括水声自监督适配、recording-level MIL、真实背景混合、双前端和 E1/E2 外测
读取器；这些应在 B3 smoke 和显存边界确认后按本计划逐项加入。

## 1. 研究目标与核心判断

本计划研究的问题不是“把当前 CNN 机械替换成更大的网络”，而是：在相同原始录音和严格隔离
协议下，**预训练 Conformer、较长上下文、录音级训练以及外部水声自监督**能否同时提高 DeepShip
四分类性能和跨水域泛化能力。

任务标签固定为：

- Cargo
- Passenger
- Tanker（项目现有目录名为 `Tank`）
- Tug

需要分别验证三个假设：

1. **长上下文假设**：10～20 s 连续波形比固定 3 s 片段包含更稳定的机械周期、工况变化和
   低频调制信息。
2. **架构假设**：在输入、上下文、预训练数据和训练目标受控时，Conformer 比强 CNN 学到更可
   迁移的水声表示。
3. **预训练假设**：通用音频/语音预训练加多站点无标签水声适配，比只用 DeepShip 从零训练
   更适合约 80M～120M 参数的大模型。

当前 MA-CNN-A 基线为 532,166 参数。已有两个 seed 的主要结果为：

| 协议 | Segment Accuracy | 主要聚合 Accuracy |
|---|---:|---:|
| `segment_level` | 97.30±0.21% | recording 98.24±0.31% |
| `recording_disjoint` | 66.20±0.49% | recording 70.19±4.08% |
| `vessel_name_disjoint` | 50.98±0.18% | vessel-name group 53.13±4.42% |

因此，后续不把已经饱和的随机片段准确率作为核心指标。主要目标是提高
`vessel_name_disjoint`、`recording_disjoint` 以及外部四分类结果。

## 2. 结论边界与公平比较原则

### 2.1 两种问题必须分开回答

**最佳系统比较**允许每个模型采用适合自己的前端和时间建模方式，但必须让模型获得相同原始
录音、相同观察时长和相同标签信息。例如在同一段 20 s 波形上：

- Conformer 直接处理 20 s 波形；
- CNN 对同一 20 s 波形产生若干短窗预测并做聚合。

该比较回答“部署时哪套完整系统更好”。

**架构因果比较**则必须控制输入表示、上下文、无标签数据、预训练目标和分类头，仅替换 CNN 与
Conformer 主干。该比较回答“提升能否归因于 Conformer 架构”。

不能用“预训练 raw Conformer”直接对比“从零训练 3 s Mel CNN”后，把全部差距归因于架构。

### 2.2 数据隔离原则

1. 沿用现有冻结的 DeepShip 隔离协议，不修改
   `configs/experiments/isolation_comparison_v1.json`。
2. Conformer 新实验使用新的版本化配置和 manifest。
3. DeepShip 的 validation/test recording 或 vessel-name group 不得参与对应协议的领域自监督。
4. 作为完全外部测试的站点、时间段、MMSI 和音频不得以无标签形式参与预训练。
5. 先按 recording、vessel/MMSI、站点和时间划分，再从波形产生动态裁剪；不得先切片再随机
   分组。
6. 测试集只在候选系统和超参数冻结后使用；不能根据外测结果反复选择模型。

## 3. 推荐主架构

第一版主线为：

```text
16 kHz mono waveform
→ 可学习的 strided Conv1d / Wav2Vec2-style feature encoder
→ 10～20 s token sequence
→ 12-layer Conformer encoder
→ Attentive Statistics Pooling
→ recording-level four-class head
```

建议起始规模，而非最终固定最优值：

| 项目 | 建议值 |
|---|---:|
| Conformer blocks | 12 |
| `d_model` | 512 |
| attention heads | 8 |
| FFN dimension | 2048 |
| depthwise convolution kernel | 31 |
| 预计参数量 | 80M～120M |
| 主训练上下文 | 20 s |
| 候选上下文 | 3/10/20/30 s |

原始采样点不能直接进入全局自注意力。可学习卷积前端先把时间分辨率降低约 200～320 倍，再由
Conformer 的卷积模块建模局部声学结构，由注意力模块建模较长周期和工况变化。

主路线采用**通用预训练权重 → 多站点无标签水声适配 → DeepShip 分阶段微调**。从零训练只作为
消融对照，不作为优先方案。

## 4. 前端与波形预处理

### 4.1 推荐处理流程

```text
读取完整 recording
→ 根据冻结协议确认 partition
→ 选定单通道并去除 DC
→ 带抗混叠滤波的 16 kHz 重采样
→ 与预训练 processor 一致的幅值归一化
→ 动态随机裁剪 3/10/20/30 s
→ 不足长度补零并生成 attention mask
→ 模型前向
```

原则：

- 不要求 Conformer 沿用当前 `64×94 log-Mel`；主分支输入原始波形。
- 不预先做通用语音降噪，因为可能删除低频线谱、包络和机械调制信息。
- 不默认固定高通滤波；低频信息可能是水声船型判别的重要组成部分。
- 归一化必须与选定预训练模型的 processor 一致，并在配置中记录。
- 训练采样按 recording 或 vessel group 平衡，避免长录音仅因切片多而支配梯度。
- 20 s 裁剪不足的短录音使用 padding 和 mask，不循环复制音频。

### 4.2 录音级训练与推理

第一阶段可使用单个动态窗口训练；随后加入 multiple-instance learning（MIL）：

- 每条 recording 随机采样 2～4 个窗口；
- 窗口共享编码器；
- 先聚合 embedding，再计算 recording-level loss；
- 避免把同一 recording 的相邻窗口视为独立标签证据。

测试覆盖完整 recording：窗口长度与模型上下文一致，默认 50% overlap；先得到窗口概率或
embedding，再聚合到 recording，最后按 vessel-name/MMSI 聚合。窗口、recording 和 vessel 三个
层级的指标均保存，但以 recording/vessel 为主要结论。

### 4.3 双前端只作为后期消融

双前端含义是同时使用：

1. 原始波形分支：保留相位、瞬态、细粒度周期和包络；
2. 低频高分辨率频谱分支：提供线谱和稳定频率结构的先验。

建议融合方式：

```text
z_raw  = pretrained_waveform_encoder(waveform)
z_spec = projection(spectral_adapter(STFT_or_subband_Mel(waveform)))
z      = LayerNorm(z_raw + sigmoid(g) * align(z_spec))
```

门控 `g` 初始化为较小值，使系统开始时接近已预训练的 raw-only 模型。必须比较 raw-only、
spectral-only 和 gated-dual；如果双前端提升不足 1 个百分点或外测不提升，则删除该分支。

## 5. 数据角色与现成来源

### 5.1 DeepShip：有标签下游训练与内部评测

本地已有 609 条 WAV、约 47.22 h。继续使用当前三种协议，主要报告
`recording_disjoint` 和 `vessel_name_disjoint`。不得通过生成更多重叠切片把片段数增长表述为
数据集扩充。

### 5.2 ONC Oceans 3.0：无标签水声领域适配

ONC 提供连续水听器录音、站点、时间和设备元数据，但原始录音通常没有对齐的船型声学标签。
它适合多站点无标签自监督，不直接作为现成四分类集。

数据选择优先级：

- 多个水听器和水域，而不是只增加同站点时长；
- 港口、近岸和相对安静站点；
- 多月份、昼夜和海况；
- 船舶活动、无船背景、多船重叠及生物/天气/施工噪声均应保留。

### 5.3 Oceanship：AIS 弱标签与长上下文外测候选

Oceanship 由 ONC 音频和船舶信息整理，提供船型、时间、坐标、速度以及增强版 MMSI/IMO 等
元数据；类别中包含 Cargo、Passenger、Tanker 和 Tug。标签主要来源于 AIS/船舶信息，属于
弱标签，不能自动等同于“该音频中只有这一条船且它是主要声源”。

下载少量样本后必须审计：

- 实际发布时长；官方仓库与论文存在 65 h/121 h 口径差异；
- 文件是否保留连续 10～20 s 以上波形；
- 每类独立 MMSI、站点和 passage 数量；
- 多船、距离、缺失 AIS 和类别映射质量；
- 是否与拟用于 ONC 自监督的数据在站点和时间上重叠。

通过审计后，Oceanship 可承担弱监督适配或 E2 外测之一，但同一批音频不能同时承担两种角色。

### 5.4 PORTIA：主要的相同四分类外测

PORTIA v2 是港口多水听器数据，提供 3 s 窗口、AIS/MMSI、距离以及 tug、pilot、cargo、
passenger、support、tanker、other 等类别。主外测只从 `Single-Vessel Classification` 中保留：

- cargo
- passenger
- tanker
- tug

排除 pilot、support、other、多船窗口、缺失 MMSI 和无效音频。PORTIA 全体音频不得参与主模型
预训练或调参。

### 5.5 不纳入核心方案的数据

ShipsEar 标签体系与 DeepShip 不一致，本计划删除原 E3，不把它纳入核心实验。只有当研究目标
扩展为“跨任务通用水声表示”时，才重新考虑独立线性探针实验。

QiandaoEar22 的多目标和背景数据可用于后续压力测试或真实背景混合，但不是第一阶段必要数据。

## 6. 两级外部评测设计

### 6.1 E1：PORTIA-4 短窗口零样本外测

目标：在完全相同的 3 s 外部波形和四分类标签下，比较 CNN 与 Conformer 的跨水域、跨设备
泛化。

协议：

1. 合并 PORTIA 相关 manifest，按 `primary_mmsi` 分组，不按窗口随机划分。
2. 按四类分层：20% MMSI 作为管线开发集，80% 作为封存测试集。
3. 外测开发集只用于检查读取、标签映射和聚合代码；模型和超参数仍根据 DeepShip validation
   决定。
4. 固定一个有效水听器通道，例如 channel 0；CNN 与 Conformer 获得相同波形。
5. 两个模型均输出 DeepShip 的四个类别，不在 PORTIA 上训练新分类头。
6. 先按窗口预测，再按 passage（若可恢复）和 MMSI 聚合。

主要指标为 MMSI-level macro-F1 和 balanced accuracy；同时报告窗口指标、每类 recall、距离
分箱结果及配对 bootstrap 置信区间。

PORTIA 公开文件是固定 3 s 窗口。除非元数据审计证明窗口连续、无缺口且允许可靠拼接，否则不
把它用于 10～20 s 长上下文结论。

### 6.2 E2：连续四分类长上下文外测

目标：验证完整 10～20 s Conformer 系统能否泛化到预训练和微调阶段均未见的水域、水听器和
物理船舶。

优先从 Oceanship 筛选；若连续性或标签质量不足，则从完全留出的 ONC 站点和历史 AIS 构建。

最低协议：

- 目标船属于 Cargo、Passenger、Tanker、Tug 之一；
- 测试站点从未参与无标签自监督；
- 外测 MMSI 从未参与弱监督训练；
- 以一次连续船舶经过（passage）为基本样本；
- 每个 passage 保留 60～180 s 连续音频；
- 用预先固定的距离、多船排除和最低航速规则筛选；
- 人工审核测试 passage 的 AIS 轨迹、频谱和显著异常，但不根据模型预测删样本。

推荐规模：

- 至少 2 个完全留出的站点；
- 每站点每类至少 20 个不同 MMSI；
- 总计约 160 个独立 MMSI；
- 每个 MMSI 保留 2～5 min，总音频约 8～15 h。

资源不足时的最低可接受方案为 1 个新站点、每类至少 30 个 MMSI、合计至少 120 个 MMSI。

完整系统比较时，CNN 和 Conformer 都获得同一段 20 s 原始波形：Conformer 直接编码 20 s，
CNN 在同一 20 s 内产生短窗预测并聚合。另报告同为 3 s 输入的受控比较，以区分长上下文收益和
架构收益。

## 7. 自监督和弱监督数据要求

### 7.1 自监督正样本

优先级从低到高：

1. 同一时刻波形的两种低成本增强；
2. 同一 recording 的相邻窗口；
3. 同一 MMSI 同一 passage 的不同距离窗口；
4. 同一 MMSI 在不同日期、距离、站点或水听器上的录音。

第 4 类最能迫使模型保留船舶相关信息并降低站点、距离和设备特征。

普通对比学习可能把未知的同一船舶当成负样本，因此优先考虑不依赖显式负样本的 teacher-student
masked prediction、VICReg、BYOL 或 data2vec 风格目标。最终目标在小规模数据上先验证，再扩大
数据量。

### 7.2 低成本增强

本计划不把高成本水声传播仿真作为前置条件。首轮仅使用：

- 随机增益；
- 随机裁剪和小幅时间平移；
- 平滑低幅度 EQ 或轻量 FIR；
- 来自真实水听器背景段的噪声混合；
- 时间/token mask；
- 相邻窗口或同 MMSI 一致性约束。

语音式强混响、显著 pitch shift 和未经验证的物理信道模拟只作为负对照或后续研究，不默认加入
主训练。

## 8. 实验矩阵

全部实验采用“单变量顺序消融”，不做所有因素的笛卡尔积。探索阶段先用一个 seed 和较短训练，
只有候选配置进入正式多 seed。

### 8.1 最佳系统线 B

| ID | 配置 | 唯一新增因素 | 回答的问题 |
|---|---|---|---|
| B0 | 当前 MA-CNN-A，3 s log-Mel | 无 | 已知基线 |
| B1 | 长上下文 CNN，20 s 内短窗聚合 | 上下文/聚合 | 提升是否仅来自更长观测 |
| B2 | raw Conformer，20 s，从零训练 | 架构与 raw 前端 | DeepShip 是否足够训练大模型 |
| B3 | 通用预训练 raw Conformer，20 s | 通用预训练 | 预训练是否是关键因素 |
| B4 | B3＋多站点水声自监督适配 | 领域适配 | 水声无标签数据是否改善泛化 |
| B5 | B4＋recording-level MIL/ASP | 训练与任务粒度对齐 | 录音级目标是否改善聚合性能 |
| B6 | B5＋低成本增强/真实背景混合 | 鲁棒性增强 | 噪声与外测是否改善 |
| B7 | B6＋门控频谱支路 | 双前端 | 频谱先验是否提供额外收益 |

B7 为累计路线；如果某一步外测不提升或显著增加复杂度，则回退到上一步作为最终系统。

### 8.2 架构因果线 C

| ID | 输入与上下文 | 架构 | 预训练 |
|---|---|---|---|
| C0 | 相同 128-bin 频谱、相同上下文 | 强 CNN/ResNet18 | 无 |
| C1 | 相同 128-bin 频谱、相同上下文 | Conformer | 无 |
| C2 | 相同频谱、数据和 SSL 目标 | 强 CNN/ResNet18 | 水声 SSL |
| C3 | 相同频谱、数据和 SSL 目标 | Conformer | 水声 SSL |

`C1-C0` 和 `C3-C2` 才是可以归因于架构的差值。B 系列用于寻找最佳系统，C 系列用于支撑论文
中的因果解释。

### 8.3 上下文消融 T

对选定的预训练 raw Conformer 运行 3/10/20/30 s；CNN 至少运行 3 s 和在相同观察时长内的聚合
版本。如果 20 s 相对 10 s 的主要指标提升不足 1 个百分点，则选用 10 s，避免无效增加显存和
延迟。

### 8.4 微调策略消融 F

| ID | 策略 |
|---|---|
| F0 | 冻结 encoder，只训练 pooling/head |
| F1 | 解冻最后 3～4 个 Conformer blocks 和 adapter |
| F2 | 全量微调，encoder 使用较小学习率 |

预计 F1 是 DeepShip 小数据下的稳健起点；F2 只有在严格验证集持续改善时保留。

### 8.5 自监督数据消融 D

| ID | 无标签水声数据 | 目的 |
|---|---|---|
| D0 | 不做水声适配 | 通用预训练基线 |
| D1 | 100 h，单站点 | 单站点数据量收益 |
| D2 | 100 h，至少 4 个站点 | 在相同时长下测站点多样性 |
| D3 | 500 h，至少 4 个站点 | 测数据规模和收益饱和 |
| D4 | D3＋MMSI/相邻窗口一致性 | 测弱标签结构化正样本 |

实际执行先做 10～20 h smoke，再做 D1/D2。只有 100 h 结果有效才扩展到 200～500 h。

### 8.6 标签效率消融 L

在选定 CNN 和 Conformer 上使用 DeepShip 训练数据的 10%/25%/50%/100%。子集按 recording 或
vessel group 分层选取，不能随机抽三秒片段。该实验判断预训练 Conformer 是否能以更少标签达到
CNN 全量标签水平。

## 9. 指标、统计与成功标准

### 9.1 主要指标

- DeepShip `vessel_name_disjoint`：vessel-name-group macro-F1 和 Accuracy；
- DeepShip `recording_disjoint`：recording macro-F1 和 Accuracy；
- E1：MMSI-level macro-F1、balanced accuracy；
- E2：passage-level 与 MMSI-level macro-F1；
- 每类 recall、混淆矩阵、距离/站点分层结果；
- 参数量、训练显存、推理时延和每小时音频处理耗时。

### 9.2 正式运行和置信区间

- 探索阶段：固定 seed 42；
- 正式候选：建议 seed 42/43/44/45/46；
- 在相同测试 group 上做 paired bootstrap，报告模型差值的 95% 置信区间；
- 测试集选择、阈值和聚合规则在正式运行前冻结。

当前 vessel-name test 约 32 个 group，一个 group 约对应 3.125 个 Accuracy 百分点，因此小于
3 个百分点的单次变化不能视为稳定证据。

### 9.3 预先定义的成功标准

最终系统至少满足：

1. DeepShip vessel-name-group 主要指标比 B0 提高至少 5 个百分点；
2. DeepShip recording-level 主要指标提高至少 3 个百分点；
3. E1 和 E2 中至少一个 macro-F1 提高 5 个百分点，另一个不明显退化；
4. 四类中至少三类 recall 提高，不能仅由单一大类驱动；
5. paired bootstrap 的模型差值 95% 置信区间下界高于零，或在样本量不足时至少大部分
   bootstrap 重采样差值为正；
6. 低标签量下表现出比 CNN 更缓慢的性能下降。

## 10. 实验前预期与可得结论

下表是研究假设和工程目标，不是已经得到的结果，也不能在论文中写成实测值。

| 对比 | 合理预期 | 若成立可得结论 |
|---|---:|---|
| B1 vs B0 | 严格指标 +1～4 pp | 长上下文本身有效 |
| B2 vs B1 | -3～+2 pp | 大 Conformer 从零训练未必优于 CNN |
| B3 vs B2 | +4～10 pp | 通用预训练是主要收益来源 |
| B4 vs B3 | DeepShip +2～6 pp；外测 +4～10 pp | 水声领域适配主要改善泛化 |
| B5 vs B4 | recording/vessel +1～3 pp | 录音级训练目标有效 |
| B6 vs B5 | 内部 0～2 pp；噪声/外测 +2～5 pp | 真实背景和低成本增强改善鲁棒性 |
| B7 vs B6 | -1～+2 pp | 双前端收益不确定，需据实保留或删除 |
| D2 vs D1 | 外测 +2～6 pp | 多站点多样性比同站点时长更重要 |

结果解释规则：

- 只有 B1 提升：收益来自长上下文，不能声称 Conformer 架构更强。
- B3 明显优于 B2，但 C3 约等于 C2：收益主要来自预训练。
- C3 明显优于 C2 且 E1/E2 同时改善：支持 Conformer 表示更可迁移。
- B4 在 DeepShip 基本不变但外测提高：支持领域适配改善跨环境泛化。
- B7 只提高 DeepShip 而外测不提高：频谱支路增加了数据集拟合，应删除或降级为可选模块。
- Segment accuracy 很高而 vessel/MMSI 不变：不能声称真实泛化改善。

保守目标区间为：vessel-name-group Accuracy 从当前约 53.13% 提升到 61%～68%，recording
Accuracy 从约 70.19% 提升到 76%～82%。该区间仅用于资源规划，不是承诺结果。

## 11. 下载量、磁盘和数据获取顺序

### 11.1 大小估算

16 kHz、单通道 PCM16 约为 115 MB/h。ONC 原始数据若为 64～96 kHz、24-bit、单通道，约为
0.69～1.04 GB/h。实际压缩率和站点产品格式会改变下载量。

| 阶段 | 数据 | 预计下载量 | 16 kHz PCM16 处理后 |
|---|---|---:|---:|
| 代码原型 | 现有 DeepShip＋预训练权重 | 所选 large checkpoint 约 2.5 GB | 无新增外部音频 |
| SSL smoke | ONC 10～20 h | 约 7～20 GB | 约 1～2.3 GB |
| 第一轮有效适配 | ONC 100 h | 约 70～105 GB | 约 11.5 GB |
| 推荐正式适配 | ONC 200 h | 约 140～210 GB | 约 23 GB |
| 规模上限消融 | ONC 500 h | 约 350～520 GB | 约 58 GB |

PORTIA v2 完整压缩音频约 75 GB；官方建议在同时保留压缩包和解压 WAV 时准备至少约 150 GB。
若再保存 16 kHz 单通道副本，处理后约 9 GB。批量处理应尽量边解压、边选通道和重采样，避免
长期保留多份原始副本。

第一轮新增下载量控制在 10～20 GB。第一套有说服力的正式实验约需 150～250 GB 下载量；加入
完整 PORTIA、处理缓存、模型权重和多个 checkpoint 后，建议外置盘至少保留 500 GB 可用空间。
若执行 500 h ONC 消融，建议准备约 1 TB 可用空间。

### 11.2 不先批量下载

改代码前不需要下载全部外部数据。执行顺序为：

1. 用现有 DeepShip 实现 raw waveform、动态上下文、mask、Conformer、MIL 和完整录音推理。
2. 只下载 ONC/Oceanship/PORTIA 的 metadata、manifest 和少量真实音频样本。
3. 用 1～5 GB 样本验证采样率、通道、PCM、路径和弱标签解析。
4. 用 ONC 10～20 h 跑通自监督 smoke，检查吞吐、loss、坍缩和 checkpoint。
5. 扩展到 100 h，并比较单站点与多站点。
6. 只有收益成立时扩展至 200～500 h。
7. 候选模型和评测脚本冻结后，再下载完整 PORTIA 并运行封存外测。

### 11.3 建议外置存储布局

延续 `docs/storage_layout.md` 的原则，大文件不放入 Git 或 Dropbox：

```text
/Volumes/T7/ProjectData/Deepship/
  datasets/
    DeepShip/
    ONC/
      raw/
      processed_16k/
      metadata/
    Oceanship/
      raw/
      processed_16k/
      metadata/
    PORTIA/
      archives/
      raw/
      processed_16k/
      manifests/
  pretrained/
  ssl_checkpoints/
  runs/
  cache/
```

Git 只保存小型 manifest、配置、指标、图表和文档；原始音频、预处理副本、预训练权重和完整
checkpoint 留在外置盘或训练服务器。

## 12. 实施阶段与验收

### 阶段 0：冻结方案和命名

- 新建 Conformer 实验配置，不修改现有 frozen isolation 配置；
- 固定标签映射、主要指标、上下文候选、seed 和成功标准；
- 为 E1/E2 定义独立版本和 manifest schema。

验收：仅凭配置和本文可明确每个实验改变了哪个变量。

### 阶段 1：DeepShip raw Conformer 基线

- 实现统一 waveform dataset 和动态 crop；
- 接入预训练 Conformer；
- 跑通 B2/B3、F0/F1/F2 和 3/10/20 s 小规模实验；
- 实现完整 recording 推理与聚合。

验收：现有三个 DeepShip 协议均无 group 泄漏；训练、resume、推理和指标可复现。

### 阶段 2：小规模水声自监督

- 下载并审计 10～20 h 多站点 ONC；
- 比较至少一种 masked/teacher-student 目标与一种 VICReg 类目标；
- 检查表示方差、协方差、下游线性探针和训练稳定性。

验收：无坍缩；B4 至少在严格 validation 或低标签实验上显示一致正向趋势。

### 阶段 3：100～200 h 领域适配和核心消融

- 执行 D1/D2，确定站点多样性的价值；
- 运行 B4/B5/B6；
- 运行 C0～C3 架构控制；
- 候选系统进入正式多 seed。

验收：形成预训练、上下文、架构和录音级目标四类可独立解释的结论。

### 阶段 4：冻结外部评测

- 下载 PORTIA metadata，建立 PORTIA-4 MMSI manifest；
- 审计 Oceanship 连续性和四类弱标签；
- 冻结 E1 开发/测试和 E2 站点/MMSI/passage 清单及哈希；
- 人工检查标签异常但不依据模型输出筛数据。

验收：E1/E2 与全部训练和 SSL 数据在音频、MMSI、站点和规定时间范围上无交集。

### 阶段 5：最终评测和报告

- 冻结代码 commit、配置、checkpoint 选择规则和聚合规则；
- 运行 DeepShip 正式 seed 与 E1/E2；
- 生成 paired bootstrap、混淆矩阵、距离/站点分层和资源报告；
- 对照第 9.3 节成功标准给出支持或否定结论。

验收：所有结论均能追溯到唯一数据 manifest、代码 commit、配置、模型权重和输出目录。

## 13. 主要风险与停止条件

| 风险 | 处理方式/停止条件 |
|---|---|
| DeepShip 太小导致从零 Conformer 过拟合 | 只把 scratch 作为对照；转向预训练和部分解冻 |
| ONC 与 DeepShip 站点相近造成隐性域重合 | 多站点训练；E2 站点完全留出 |
| Oceanship 弱标签或连续性不足 | 不作为 E2；改用留出 ONC＋AIS 自建 |
| PORTIA 只有 3 s | 仅用于 E1；不据此评价长上下文收益 |
| AIS 最近船不是主要声源 | 多船/距离筛选和人工审核；报告弱标签限制 |
| 双前端复杂但收益小 | 外测不足 1 pp 或置信区间跨零即删除 |
| 20 s 相比 10 s 无收益 | 选择 10 s，停止扩大上下文 |
| 100 h 多站点 SSL 无稳定收益 | 暂停 500 h 下载，先复核目标和采样策略 |
| 外测被用于多轮模型选择 | 重建新的封存测试或严格限制结论为外测开发结果 |

## 14. 参考资料

1. [DeepShip 原始数据集论文](https://www.sciencedirect.com/science/article/pii/S0957417421007016)
2. [Conformer 原始论文](https://arxiv.org/abs/2005.08100)
3. [Hugging Face Wav2Vec2-Conformer 文档](https://huggingface.co/docs/transformers/main/model_doc/wav2vec2-conformer)
4. [水声 Conformer＋VICReg 泛化表示研究](https://ir.cwi.nl/pub/35751/35751.pdf)
5. [上述研究的官方代码 UATR](https://github.com/hildeingvildhummel/UATR)
6. [ONC Oceans 3.0 数据入口](https://www.oceannetworks.ca/data/)
7. [Oceanship 官方仓库](https://github.com/lizeyujack/oceanship)
8. [PORTIA v2 数据与 manifest](https://zenodo.org/records/21409381)
9. [QiandaoEar22 数据说明](https://arxiv.org/abs/2406.04353)
10. [项目现有隔离训练路线](strict_isolation_training_plan.md)
11. [项目数据与贡献边界](handbook/01_项目数据与贡献边界.md)
12. [项目存储布局](storage_layout.md)
