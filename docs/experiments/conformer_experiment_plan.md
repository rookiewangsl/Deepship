# DeepShip Conformer 实验、数据与外部评测计划

最后更新：2026-08-27
状态：B3 工程基线、服务器环境、预训练权重和无梯度真实前向均已验证；尚未执行反向传播或训练，
不代表已有性能结果

## 0. 当前实施状态

第一版可运行基线已经落地，Linux/RTX 4070 的环境准备、预检、smoke 和正式运行顺序见
[`conformer_baseline_linux.md`](../guides/conformer_baseline_linux.md)。服务器已缓存约 2.4 GB 的
`facebook/wav2vec2-conformer-rel-pos-large`，并在一条真实 20 s DeepShip 波形上完成无梯度前向：
输入为 `[1, 320000]`，输出为 `[1, 4]`，FP16 峰值显存约 2.8 GiB。尚未运行反向、优化器更新或
正式训练。

已实现范围：

- 使用原有冻结隔离 manifest，不在训练时重新划分数据；
- 以 3 s anchor 中心扩展 3/10/20/30 s 原始波形上下文；
- 官方预训练 Wav2Vec2-Conformer large、ASP 四分类头和分阶段解冻；
- checkpoint/resume、segment/recording/vessel-group 三级评测和可复现实验记录；
- 服务器 Python/CUDA/RTX 4070、DeepShip 路径、冻结 manifest 和 Hugging Face 缓存预检；
- PORTIA 官方四类原始标注包含 18,599 个窗口；排除缺失/无效 MMSI 或距离字段后，冻结
  MMSI-disjoint manifest 包含 15,114 个窗口，development/test MMSI 无交叉；
- PORTIA 完整音频正在服务器后台下载，完成后自动 MD5 校验、分卷 ZIP 测试、解压和 WAV 索引。

尚未实现范围仍包括水声自监督适配、recording-level MIL、真实背景混合、双前端和 E1/E2 模型
推理读取器；这些只在前一条研究路线出现支持证据后按本计划加入。

## 1. 研究目标与核心判断

本计划研究的问题不是“把当前 CNN 机械替换成更大的网络”，而是研究通用自监督声学模型从
通用领域到水声领域、再到船舶分类任务的逐级适配，以及这种适配是否带来真正的外部泛化：

```text
通用语音自监督预训练
→ 多站点无标签水声领域适配
→ DeepShip 有标签任务微调
→ PORTIA 未见水域与 MMSI 零样本评测
```

其中，水声自监督解决“领域适配”，DeepShip 监督微调解决“任务适配”，PORTIA 负责判断提升是
数据集内拟合还是真实跨域泛化。Conformer 是承载这条研究路线的基础模型，不把参数规模本身当作
贡献。

任务标签固定为：

- Cargo
- Passenger
- Tanker（项目现有目录名为 `Tank`）
- Tug

需要分别验证五个假设：

1. **通用迁移假设**：语音自监督预训练即使没有见过水声，也比相同 Conformer 随机初始化提供
   更有效的船舶声学表示。
2. **任务微调假设**：DeepShip 小样本条件下，冻结底层并只调整高层比全量微调更能保留通用
   表征并抑制过拟合。
3. **领域适配假设**：从同一通用 checkpoint 出发，先在多站点无标签 ONC 水声上继续自监督，
   再用 DeepShip 监督微调，主要改善跨水域泛化而非只提高训练集拟合。
4. **长上下文假设**：10～20 s 连续波形比固定 3 s 片段包含更稳定的机械周期、工况变化和
   低频调制信息。
5. **架构假设**：在输入、上下文、预训练数据和训练目标受控时，Conformer 比强 CNN 学到更可
   迁移的水声表示。

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

## 3. 当前主架构与研究载体

第一版主线为：

```text
16 kHz mono waveform
→ 7-layer Wav2Vec2 strided Conv1d feature encoder
→ 10～20 s token sequence
→ 24-layer Wav2Vec2-Conformer encoder（relative position）
→ Attentive Statistics Pooling
→ four-class head
```

当前实际 checkpoint 配置：

| 项目 | 当前值 |
|---|---:|
| checkpoint | `facebook/wav2vec2-conformer-rel-pos-large` |
| Conformer blocks | 24 |
| `hidden_size` | 1024 |
| attention heads | 16 |
| FFN dimension | 4096 |
| 总参数量（含当前分类头） | 619,353,477 |
| `last-4` 可训练参数 | 101,701,381 |
| 主训练上下文 | 20 s |
| 候选上下文 | 3/10/20/30 s |

原始采样点不能直接进入全局自注意力。可学习卷积前端先把时间分辨率降低约 200～320 倍，再由
Conformer 的卷积模块建模局部声学结构，由注意力模块建模较长周期和工况变化。

该 checkpoint 在 16 kHz、960 h LibriSpeech 语音上自监督预训练，不是水声专用模型。主路线采用
**通用预训练权重 → 多站点无标签水声继续自监督 → DeepShip 分阶段微调**。同时保留“通用权重
直接微调 DeepShip”的基线，从零训练只作为参数规模与预训练价值的控制，不作为优先方案。

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

截至 2026-08-27 的标注预审结果：四个目标类别原始共有 18,599 个窗口；要求有效 MMSI、距离和
唯一窗口 ID 后，冻结 manifest 保留 15,114 个窗口。独立 MMSI 数分别为 Cargo 44、Passenger
21、Tank 5、Tug 6；80% 封存测试中分别为 35、16、4、4。因此 Tank/Tug 的 MMSI-level 结论
必须报告 bootstrap 区间，并明确小组数限制，不能把大量窗口当作同等数量的独立船舶样本。

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
3. 外测开发集用于检查读取、标签映射和聚合代码，并可用于判断是否存在值得研究的领域偏移；
   学习率、微调深度、checkpoint 和其他超参数仍只根据 DeepShip validation 决定。
4. 固定一个有效水听器通道，例如 channel 0；CNN 与 Conformer 获得相同波形。
5. 两个模型均输出 DeepShip 的四个类别，不在 PORTIA 上训练新分类头。
6. 先按窗口预测，再按 passage（若可恢复）和 MMSI 聚合。
7. 80% 封存测试集在 B3、条件触发的 B4 及全部聚合规则冻结后只运行一次，不能用于决定是否
   启动 ONC、选择 SSL 目标或调整模型。

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
| B4 | 通用 checkpoint → 多站点水声自监督 → 与 B3 相同的 DeepShip 微调 | 领域适配 | 水声无标签数据是否改善泛化 |
| B5 | B4＋recording-level MIL/ASP | 训练与任务粒度对齐 | 录音级目标是否改善聚合性能 |
| B6 | B5＋低成本增强/真实背景混合 | 鲁棒性增强 | 噪声与外测是否改善 |
| B7 | B6＋门控频谱支路 | 双前端 | 频谱先验是否提供额外收益 |

B4 不是在已经完成监督微调的 B3 权重上继续自监督，而是从相同通用 checkpoint 重新开始领域
自监督，然后使用与 B3 完全相同的 DeepShip 监督微调协议。B5～B7 为条件分支；如果某一步外测
不提升或显著增加复杂度，则回退到上一步作为最终系统。

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

微调研究围绕“需要改动多少通用表征”展开。第一轮不叠加 LoRA、增强或双前端。

| ID | 策略 | 作用 |
|---|---|---|
| F0 | 冻结卷积前端和 24 层 Conformer，只训练 ASP/head | 判断通用表示能否直接浅层迁移 |
| F1a | 解冻最后 2 个 Conformer blocks | 保守任务适配 |
| F1b | 解冻最后 4 个 Conformer blocks | 当前 B3 主方案 |
| F1c | 解冻最后 8 个 Conformer blocks | 检查更深任务适配是否必要 |
| F2 | 冻结卷积 feature encoder，微调全部 Conformer blocks | 调整全部高层时序表示但保留底层波形特征 |
| F3 | 卷积前端和 Conformer 全量微调 | 检查底层语音声学特征是否也需重构 |
| F4 | LoRA/Adapter 等参数高效微调 | 仅在 F1 过拟合或 F2/F3 显存受限时加入 |

执行顺序为 `F0 → F1b → 根据结果选择 F1a/F1c → F2 → F3`。F4 不是核心必做项。预计 F1b 是
DeepShip 小数据下的稳健起点；F2/F3 只有在严格 validation 持续改善时保留。所有 backbone 参数
使用远低于 pooling/head 的学习率，避免快速破坏预训练表示。

### 8.5 自监督数据消融 D

| ID | 无标签水声数据 | 目的 |
|---|---|---|
| D0 | 不做水声适配 | 通用预训练基线 |
| D1 | 100 h，单站点 | 单站点数据量收益 |
| D2 | 100 h，至少 4 个站点 | 在相同时长下测站点多样性 |
| D3 | 500 h，至少 4 个站点 | 测数据规模和收益饱和 |
| D4 | D3＋MMSI/相邻窗口一致性 | 测弱标签结构化正样本 |

实际执行先做 10～20 h smoke，再做 D1/D2。只有 100 h 结果有效才扩展到 200～500 h。

### 8.6 通用权重与水声适配权重的公平微调

B3 与 B4 必须使用相同的 DeepShip split、上下文、pooling/head、优化器、训练预算和模型选择规则。
先将 B3 选出的最佳微调深度直接用于 B4，再额外检查水声适配后是否需要更浅的解冻：

- 如果 B4 在 F0/F1a 下已达到 B3 的 F1b/F1c，说明领域自监督承担了大部分语音—水声适配；
- 如果 B4 仍需更深解冻，说明 ONC 自监督只提供了初始化改善，没有完成任务相关适配；
- 如果 B4 只提高 DeepShip 而不提高 PORTIA，不能声称领域自监督改善了泛化。

### 8.7 标签效率消融 L

在选定 CNN 和 Conformer 上使用 DeepShip 训练数据的 10%/25%/50%/100%。子集按 recording 或
vessel group 分层选取，不能随机抽三秒片段。该实验判断预训练 Conformer 是否能以更少标签达到
CNN 全量标签水平。

### 8.8 路线进入与停止条件

| 已观察到的证据 | 下一步 | 当时允许形成的中间结论 |
|---|---|---|
| B3 不优于 CNN，且预训练不优于 B2 | 停止 ONC 扩容，先复核模型/任务匹配 | 当前通用预训练 Conformer 无明显价值 |
| B3 优于 B2，但相同输入架构控制无提升 | 保留预训练路线，不强调 Conformer 架构 | 收益主要来自预训练而非架构 |
| 10/20 s 优于 3 s，且严格 group 指标同步改善 | 保留最佳上下文 | 中长时船舶声学动态有效 |
| 只在 DeepShip 提升，PORTIA development 不提升 | 启动 10～20 h ONC 领域自监督 | 存在明显语音—水声或水域域偏移 |
| 10～20 h ONC 显示稳定外测正趋势 | 扩展 100 h 多站点 | 水声领域自监督值得扩大 |
| 100 h 多站点仍无稳定收益 | 停止 200～500 h 扩容 | 当前 SSL 目标或数据选择不成立 |
| 原始波形模型在线谱相关类别持续落后 Mel CNN | 才尝试 B7 双前端 | 显式谱先验可能是缺失因素 |
| 增强只改善内部、不改善外测 | 删除对应增强 | 增强增加数据集拟合而非泛化 |

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

### 10.1 项目的论文叙事

推荐主叙事不是“用更大的 Conformer 替换 CNN”，而是：

> 现有水声船舶分类容易受到片段级数据关联影响。先建立 recording/vessel-disjoint 的可信评测，
> 再研究通用语音自监督 Conformer 如何经过任务微调迁移到小样本水声分类，拆解预训练、微调深度、
> 长上下文和架构的贡献；最后在未见水域与 MMSI 的 PORTIA 上检验泛化。若通用预训练仍受领域偏移
> 限制，再用多站点无标签 ONC 水声继续自监督，验证“通用预训练 → 领域适配 → 任务适配”的层级
> 迁移是否成立。

核心贡献优先收敛为四项：

1. 严格的 recording/vessel 身份隔离和外部 MMSI 评测；
2. 通用语音自监督模型向水声船舶分类的迁移证据；
3. 预训练、微调深度、长上下文与架构贡献的可解释拆解；
4. DeepShip 内部性能与 PORTIA 外部泛化之间的完整证据链。

只有 ONC 路线实际改善封存外测时，才增加第五项贡献：多站点无标签水声领域适配。

### 10.2 可能形成的最终结论

**最强结论**：通用预训练 Conformer 在严格 DeepShip 协议和 PORTIA MMSI 外测上均优于 CNN；
长上下文提供额外收益，ONC 领域自监督进一步改善跨水域泛化。此时可以形成完整的层级迁移故事。

**中等结论**：Conformer 在 DeepShip 上优于 CNN，但直接跨域无优势；ONC 领域适配后 PORTIA
才改善。此时结论是通用预训练提供基础表示，但水声领域自监督是获得外部泛化的必要环节。

**迁移有效但架构不占优**：B3 明显优于 scratch，但相同输入的 CNN/Conformer 控制差异很小。
此时应把论文重点放在预训练与适配，而不是声称 Conformer 架构本身更强。

**长上下文故事**：10～20 s 在严格 group 指标上稳定优于 3 s，但架构差异有限。此时结论是船舶
中长时动态具有判别价值，系统收益主要来自上下文和聚合策略。

**负结果但仍可发表的结论**：严格隔离和外测下，大型语音预训练 Conformer 未稳定优于 CNN，
或 ONC 自监督无法修复域偏移。此时可以据实说明模型规模和端到端表示不天然带来水声泛化，局部
频谱结构、预训练领域匹配或数据身份隔离比增加模型容量更重要。

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

### 11.2 按证据扩充数据

当前 PORTIA 已开始完整下载；这不意味着提前使用外测调参。后续数据扩充按研究证据决定：

1. 用现有 DeepShip 和通用预训练权重建立 B3，完成冻结、部分解冻、上下文和 scratch 控制。
2. PORTIA 下载与标注审计可并行完成，但封存 test 在 B3/B4 候选及聚合规则全部固定前保持不可见。
3. 先用冻结 B3 运行 PORTIA development，诊断内部提升是否可能受外部领域偏移限制。
4. 只有“B3 内部有效、PORTIA development 显示外部不足”时，下载 ONC 10～20 h 多站点音频并
   运行 SSL smoke。
5. 只有小规模 SSL 在严格 validation 或外测 development 显示稳定趋势，才扩展到 100 h，并比较
   单站点与多站点。
6. 100 h 多站点结果成立后才扩展到 200 h；500 h 仅作为明确存在规模收益时的上限消融。
7. Oceanship/连续 ONC E2 仅在需要验证 10～20 s 长上下文跨域结论时构建，不作为默认必做数据。

### 11.3 建议外置存储布局

延续 [`storage_layout.md`](../guides/storage_layout.md) 的原则，大文件不放入 Git 或 Dropbox：

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
- 按 F0 → F1b → 条件追加 F1a/F1c/F2/F3 的顺序筛选微调深度，并比较 3/10/20 s；
- 实现完整 recording 推理与聚合。

验收：现有三个 DeepShip 协议均无 group 泄漏；训练、resume、推理和指标可复现。

### 阶段 2：冻结通用迁移基线并诊断外部开发集

- 在 DeepShip validation 上选择 B3 的微调深度与上下文；
- 完成 scratch、冻结、部分解冻和必要的全量微调控制；
- 冻结 checkpoint 选择和聚合规则；
- 审计 PORTIA 音频并只运行 development，判断是否存在值得研究的领域偏移；封存 test 不运行。

验收：能够区分预训练、微调深度、上下文与架构贡献，并决定是否有证据启动 ONC 领域适配。

### 阶段 3：条件触发的小规模水声自监督

仅当 B3 内部有效但 PORTIA development 显示外部不足时进入：

- 下载并审计 10～20 h 多站点 ONC；
- 比较至少一种 masked/teacher-student 目标与一种 VICReg 类目标；
- 检查表示方差、协方差、下游 probe、训练稳定性及 B4 相对 B3 的外测 development 变化。

验收：无坍缩；B4 至少在严格 validation 或外测 development 上显示一致正向趋势，否则停止扩容。

### 阶段 4：100～200 h 领域适配与条件分支

- 执行 D1/D2，确定站点多样性的价值；
- 只有 100 h 有效才扩展到 200 h；
- 根据错误证据决定是否加入 MIL、低成本增强或双前端；
- 必要时构建与 ONC SSL 站点/时间隔离的连续 E2。

验收：水声领域适配、站点多样性和所有附加模块都有独立外测证据；无收益分支已停止并删除。

### 阶段 5：最终评测和报告

- 冻结代码 commit、配置、checkpoint 选择规则和聚合规则；
- 运行 DeepShip 正式多 seed、E1，以及确有必要时的 E2；
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
11. [项目存储布局](../guides/storage_layout.md)
