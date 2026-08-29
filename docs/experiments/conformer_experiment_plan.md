# DeepShip 预训练 Conformer、轻量全局注意力与外部评测计划

最后更新：2026-08-29
状态：B3 工程基线与第二版组级选模管线已验证。第二版 F0、F1b、F0-S1、F0-S2 和 F1a 已完成。
F1b 明显过拟合，不进入 last-8。S1 显著改善 recording、但未改善 vessel；S2 将主优化方向转为
未见船名后，validation vessel macro-F1 提高到 0.5625，并且 recording 未低于 F0-S0，因此通过
预设点估计幅度门。F1a last-2＋S2 将 vessel macro-F1 提高到 0.6435、recording 保持 0.4380，
因此作为当前迁移候选，但单 seed vessel 配对区间仍跨零。3 s G0/G0-C/G1 已完成，G1 未优于两个
对照；当前执行共同 20 s 的 L20 受控复核，判断 3 s 近全局卷积感受野是否掩盖了注意力价值。
test 继续封存，不参与开发期选模。大型 scratch Conformer 已从必做矩阵删除；
轻量全局注意力是与通用预训练迁移并行的核心研究线，而不是 raw Conformer 失败后的临时补救

## 0. 当前实施状态

第一版可运行基线已经落地，Linux/RTX 4070 的环境准备、预检、smoke 和正式运行顺序见
[`conformer_baseline_linux.md`](../guides/conformer_baseline_linux.md)。服务器已缓存约 2.4 GB 的
`facebook/wav2vec2-conformer-rel-pos-large`，并在一条真实 20 s DeepShip 波形上完成无梯度前向：
输入为 `[1, 320000]`，输出为 `[1, 4]`，FP16 无梯度峰值显存约 2.8 GiB。随后完成 20 s、
last-4、BF16、关闭 checkpointing 的两批次反向 smoke；`nvidia-smi` 采样峰值约 6.7 GiB，最后
4 层和任务头均有有限梯度并发生合理幅度更新。旧 F0 的 train loss 从 1.0682 降至 0.4460、train
accuracy 从 53.25% 升至 83.06%，但 validation loss 从 1.3555 升至 2.0679，validation accuracy
仅为 49.50%。这说明主要问题是组外泛化和过度自信，而不是训练不足。旧 F0 没有
`run_complete.json`，旧 F1b 未启动，二者输出目录不复用。

第二版 F0 的最佳 validation vessel macro-F1 为 **0.4726**，vessel accuracy 为 0.4800，
recording macro-F1 为 0.4181。第二版 F1b 的最佳 checkpoint 出现在 epoch 3，validation vessel
macro-F1 为 **0.4650**，vessel accuracy 为 0.4800，recording macro-F1 为 0.4006；随后 train
loss 继续下降而 validation loss 上升到 2 以上，形成明确过拟合。以相同 validation vessel 做
50,000 次 paired bootstrap，F1b−F0 的 macro-F1 差为 −0.0076，95% 区间
[−0.1446, +0.1474]，`P(Δ>0)=0.4818`。这不能证明 F0 优于 F1b，但足以说明单 seed 下 last-4
没有观察到收益，继续解冻 last-8 缺少依据。

已实现范围：

- 使用原有冻结隔离 manifest，不在训练时重新划分数据；
- 以 3 s anchor 中心扩展 3/10/20/30 s 原始波形上下文；
- 官方预训练 Wav2Vec2-Conformer large、ASP 四分类头和分阶段解冻；
- 当前 12 GB RTX 4070 基线默认 BF16 并关闭 gradient checkpointing；可选 checkpointing 使用
  非重入实现，训练首个 optimizer step 强制验证所有可训练参数均有有限梯度；
- checkpoint/resume、segment/recording/vessel-group 三级评测和可复现实验记录；
- validation vessel/recording macro-F1 选模、独立 early-stopping `min_delta` 和 optimizer-step
  warmup＋cosine 调度；
- S1 class→recording-balanced 动态裁剪：每次裁剪由显式 seed 决定，worker 数和断点续训不改变
  窗口；每 epoch 保存 recording/vessel 暴露、重复率、吞吐和 DataLoader 等待占比；
- S2 class→vessel→recording-balanced 动态裁剪及其确定性暴露审计；
- validation/test 保持冻结 anchor，validation 可使用独立 batch size；开发期默认不读取 test，
  最终方案冻结后才显式开启；
- 服务器 Python/CUDA/RTX 4070、DeepShip 路径、冻结 manifest 和 Hugging Face 缓存预检；
- PORTIA 官方四类原始标注包含 18,599 个窗口；排除缺失/无效 MMSI 或距离字段后，冻结
  MMSI-disjoint manifest 包含 15,114 个窗口，development/test MMSI 无交叉；
- PORTIA 完整分卷音频归档已下载到服务器（约 70 GB），尚未解压；进入外测前先完成所有分卷的
  官方 MD5 校验和分卷感知的解压测试，再生成 WAV 索引。普通 `unzip -t` 对该分卷格式产生的
  offset/overlap 提示不能单独作为损坏结论。

F0-S1 最佳 checkpoint 位于 epoch 6：validation vessel macro-F1 为 **0.4799**、recording
macro-F1 为 **0.5718**。相对 F0-S0，vessel macro-F1 增加 0.0073，50,000 次同 vessel 配对
bootstrap 的 95% 区间为 [−0.1059, +0.1207]，`P(Δ>0)=0.5605`；recording macro-F1 增加
0.1537，同 recording 配对区间为 [+0.0744, +0.2358]，`P(Δ>0)=0.99998`。该 bootstrap 只描述
当前 validation group 的抽样不确定性，不包含训练 seed 不确定性。S1 没有通过 vessel +1 pp 的
幅度门，但 recording 改善明确，且 S1 下单 vessel 每轮暴露仍为 24～1165 次，因此触发 S2。

F0-S2 在 epoch 4 取得最佳 validation vessel macro-F1 **0.5625**、recording macro-F1
**0.4360**。相对 F0-S0，vessel 增加 **0.0900**，50,000 次同 vessel 配对区间为
[−0.0638, +0.2482]，`P(Δ>0)=0.8753`；recording 增加 **0.0179**，同 recording 区间为
[−0.0697, +0.1085]，`P(Δ>0)=0.6579`。相对 S1，S2 的 vessel 增加 **0.0827**，但 recording
下降 **0.1358**，且 recording 配对区间 [−0.2233, −0.0447] 完全低于零。这说明 S1/S2 的
训练分布目标确实不同：S1 强化录音等权，S2 强化船舶等权。S2 按预先冻结的点估计门通过，但
50 艘 validation vessel 的区间仍跨零，单 seed 不能写成确定性提升。

F1a 最佳 checkpoint 位于 epoch 2：validation vessel macro-F1 为 **0.6435**、recording macro-F1
为 **0.4380**。相对 F0-S2，vessel 增加 0.0810，50,000 次 paired bootstrap 的 95% 区间为
[−0.0513,+0.2292]，`P(Δ>0)=0.8849`；recording 增加 0.0020，区间为
[−0.0844,+0.0882]。它通过预设单 seed 点估计门，但不能写成确定性提升，也不再扩大解冻。

G 系列已实现 G0 数值等价重构、G0-C 局部容量对照和 G1 shared temporal axial attention。参数量分别为
532,166、679,593 和 694,057；G0-C/G1 新增参数差 8.93%，前向 FLOPs 差 8.53%，满足冻结的
10%/15% 匹配门。3 s 正式结果的 vessel macro-F1 为 **0.5822/0.5855/0.5132**，recording 为
**0.5694/0.6127/0.4910**，G1 未通过门。由于 3 s 输入经 CNN 后仅约 50 个时间位置且局部感受野
已接近全片段，当前只补共同 20 s 的 L20 三模型复核。尚未完成的是 L20 真实数据 smoke、三个
L20 正式 run、E1 推理读取器，以及仅在
证据触发时才进行的 10～20 h 水声自监督原型。recording-level MIL、真实背景混合、双前端和
大规模水声自监督不属于本项目完成条件，统一降级为未来工作。

### 0.1 从当前训练开始的执行顺序

```text
冻结新版 F0/F1b/S1 的 validation 诊断，不恢复 F1b、不运行 last-8
完成 F0-S2 vessel-balanced 动态裁剪和 validation paired bootstrap
→ S2 已通过点估计门；F1a last-2＋S2 已完成并通过单 seed 点估计门
→ 无论 F1a 结果如何，不再扩大解冻层数或运行大型 scratch
→ 冻结最低成本、validation 最好的 B3 配置并形成通用预训练迁移阶段结论
→ 3 s G0/G0-C/G1 已完成，G1 未通过；执行唯一共同 L20 复核
→ G1-L20 只有同时优于 G0-L20 与 G0-C-L20，vessel 至少 +1 pp 且 recording 最多下降 1 pp，才补多 seed
→ 两条内部研究线均冻结后进入 PORTIA development
→ 只有“通用预训练内部显示价值、但外部明显不足”时才启动 10～20 h ONC 自监督原型
→ 原型没有稳定改善即停止，不扩展到 100～500 h
```

这一顺序分别回答“大模型通用预训练能否低成本迁移”和“全局注意力本身能否改善专用水声
分类器”。大型 scratch Conformer 不再是完成条件，因为其成本高，且即使完成也不能消除 raw
前端、模型规模和优化难度等混杂因素。轻量注意力的详细结构、容量控制和决策门见
[`lightweight_global_attention_plan.md`](lightweight_global_attention_plan.md)。本项目首轮不叠加
JASA/MBAT 式身份不变目标，避免把架构收益与损失正则化混为一谈。

## 1. 研究目标与核心判断

本计划研究的问题不是“把当前 CNN 机械替换成更大的网络”，而是研究通用自监督声学模型从
通用领域到水声领域、再到船舶分类任务的逐级适配，以及这种适配是否带来真正的外部泛化：

```text
通用语音自监督预训练
→ DeepShip 有标签任务微调
→ PORTIA 未见水域与 MMSI 零样本评测
→ 若且仅若已证明通用预训练有效但存在外部域偏移，再进行小规模无标签水声领域适配
```

DeepShip 监督微调解决“任务适配”，PORTIA 负责判断提升是数据集内拟合还是真实跨域泛化。水声
自监督不是固定主线：它只用于检验已经明确的语音—水声域偏移，而且先限于 10～20 h 原型。
Conformer 是承载这条研究路线的基础模型，不把参数规模本身当作贡献。

任务标签固定为：

- Cargo
- Passenger
- Tanker（项目现有目录名为 `Tank`）
- Tug

需要分别验证五个假设：

1. **通用迁移假设**：语音自监督预训练即使没有见过水声，仍可能以有限微调成本提供有用的船舶
   声学表示；若最终不优于小型专用 CNN，则说明跨域迁移的成本/风险不具工程优势，而不是继续用
   大型 scratch 实验补救。
2. **任务微调假设**：DeepShip 小样本条件下，冻结底层并只调整高层比全量微调更能保留通用
   表征并抑制过拟合。
3. **条件领域适配假设**：只有通用预训练迁移在 DeepShip 内部接近或优于 CNN、但 PORTIA 暴露
   明显外部域偏移时，才检验从同一 checkpoint 出发的 10～20 h 多站点无标签水声继续自监督能否
   改善跨水域泛化。
4. **长上下文假设**：10～20 s 连续波形比固定 3 s 片段包含更稳定的机械周期、工况变化和
   低频调制信息。
5. **全局注意力假设**：在 log-Mel 输入、上下文、参数增量和训练预算受控时，CNN 后端的轻量
   temporal self-attention 比纯 CNN 和容量匹配的 Conv/MLP 对照获得更好的未见船名泛化。

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

**轻量全局注意力比较**必须控制 log-Mel 输入、上下文、训练数据、采样、优化器和训练预算，比较
原始 CNN、容量匹配的 Conv/MLP 对照和 CNN＋temporal self-attention。该比较回答“全局时间交互
是否提供超出参数容量的独立收益”，不把混合架构称为纯 Transformer。

不能用“预训练 raw Conformer”直接对比“从零训练 3 s Mel CNN”后，把全部差距归因于架构。
同样不能用 20 s attention 模型直接对比 3 s CNN 后，把差距归因于注意力。

### 2.2 数据隔离原则

1. 沿用现有冻结的 DeepShip 隔离协议，不修改
   `configs/experiments/isolation_comparison_v1.json`。
2. Conformer 新实验使用新的版本化配置和 manifest。
3. DeepShip 的 validation/test recording 或 vessel-name group 不得参与对应协议的领域自监督。
4. 作为完全外部测试的站点、时间段、MMSI 和音频不得以无标签形式参与预训练。
5. 先按 recording、vessel/MMSI、站点和时间划分，再从波形产生动态裁剪；不得先切片再随机
   分组。
6. 测试集只在候选系统和超参数冻结后使用；不能根据外测结果反复选择模型。
7. 不复现 JASA/MBAT 的训练测试 split。已有船名 embedding 诊断只作为身份捷径的失败分析证据。
   身份不变 objective 若未来重启，必须在最佳架构冻结后作为独立损失消融，不与 G0/G0-C/G1
   首轮同时加入。所有新增实验继续使用本项目冻结的 `vessel_name_disjoint`。

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

该 checkpoint 在 16 kHz、960 h LibriSpeech 语音上自监督预训练，不是水声专用模型。当前迁移线
采用 **通用预训练权重 → DeepShip 分阶段微调 → 条件触发的多站点无标签水声继续自监督**。
通用权重直接微调 DeepShip 是基线；不再训练同规模 scratch Conformer。该路线回答现实适配成本和
系统价值，而不是把 raw 前端、预训练、模型规模和 Transformer 架构强行拆成完全因果的对照。

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

### 4.2 组平衡动态采样

冻结 manifest 继续决定 recording/vessel 的 partition 和标签，但不再要求训练阶段枚举其中所有
3 s anchor。动态采样只作用于训练集；validation/test 使用固定、可复现并覆盖对应 recording 的
窗口，确保选模曲线和模型比较不受随机裁剪波动影响。

按顺序比较三个采样方案：

| ID | 训练采样单位 | 规则 | 主要诊断 |
|---|---|---|---|
| S0 | 固定 anchor | 当前每个 3 s anchor 扩展为 20 s 窗口 | 现有基线，可能含大量相邻重叠窗口 |
| S1 | recording-balanced | 先按类别、再等概率选 recording，并在 recording 内动态裁剪 | 长录音是否因窗口多而支配训练 |
| S2 | vessel-balanced | 先按类别、再等概率选 vessel、recording 和动态窗口 | 多录音船舶是否因样本多而支配训练 |

S1/S2 的每 epoch optimizer step 数与 S0 保持一致，使比较不混入训练预算差异。所有裁剪严格限制
在已分配的 recording 内；不足上下文的录音继续使用 padding 和 attention mask。先用最佳微调
深度做 seed 42 顺序筛选，只有改善 validation recording/vessel macro-F1 的方案进入正式多 seed。

第一轮不加入 probe、在线困难样本挖掘或 oversampling。若 S1/S2 已解决相邻窗口冗余，则不再
增加 HGRS；只有组平衡后仍观察到大量低损失重复样本时，才测试固定比例的 hard＋random 采样。

S1/S2 均已实现，仍使用每类 3,500、合计 14,000 次 draw，使 optimizer update 数与 S0 完全相同。
类内 recording 访问次数最多相差 1；不同 epoch 使用不同、但可由 seed 复现的裁剪。真实训练集预检
覆盖 399 条 recording 和 162 个 vessel，S1 下单 vessel 每轮暴露为 24～1165 次（median 48），
说明 recording 平衡没有消除多录音 vessel 的影响。S1 已显著改善 recording macro-F1，但 vessel
主指标只提高 0.73 pp，因此现在实际运行 S2，检查类内 vessel 等权是否能把录音级收益转化为未见
船名泛化收益。

### 4.3 录音级训练与推理（未来工作）

当前项目保持单个动态窗口训练和录音级聚合评测。以下 multiple-instance learning（MIL）只作为
未来工作，不进入当前实验矩阵：

- 每条 recording 随机采样 2～4 个窗口；
- 窗口共享编码器；
- 先聚合 embedding，再计算 recording-level loss；
- 避免把同一 recording 的相邻窗口视为独立标签证据。

测试覆盖完整 recording：窗口长度与模型上下文一致，默认 50% overlap；先得到窗口概率或
embedding，再聚合到 recording，最后按 vessel-name/MMSI 聚合。窗口、recording 和 vessel 三个
层级的指标均保存，但以 recording/vessel 为主要结论。

### 4.4 双前端（未来工作）

双前端不属于当前项目完成条件。若未来有独立研究需求，其含义是同时使用：

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

### 4.5 多核 CPU 数据处理与 I/O 方案

GPU 只承担模型前向、反向和损失计算；音频读取、选通道、局部解码、重采样、去 DC、归一化、
动态裁剪、padding 和 batch 预取由 DataLoader 多进程完成。后续动态采样不预先物化大量重叠窗口，
而是由 worker 按 recording 的帧范围直接读取所需片段，从源头减少磁盘占用和重复写入。

在线训练管线采用以下规则：

- `num_workers` 和 `prefetch_factor` 都进入版本化训练配置；有 worker 时启用
  `persistent_workers`，CUDA 训练启用 pinned memory 和 non-blocking 传输；
- 每个 worker 内将 PyTorch CPU kernel 限制为单线程，避免“DataLoader 进程数 × BLAS/重采样内部
  线程数”造成过度并行；并行单位是 recording/window，而不是单个算子再嵌套开满全部核心；
- 每个 worker 缓存文件头和不同源采样率的 resampler，但不缓存完整音频到内存；validation/test
  仍按固定窗口顺序读取，保证可复现；
- 不能直接把 `num_workers` 设成服务器全部逻辑核心。先在代表性 20 s 输入上比较
  `4/8/16/32` workers 和 `prefetch_factor=2/4`，记录稳定 samples/s、GPU 空闲比例、CPU 利用率、
  内存和磁盘吞吐，选择达到吞吐平台期的最小配置；
- 正式训练期间不同时启动高并发离线转码，以免与训练 DataLoader 争抢 `/home` 的 I/O。manifest
  审计、PORTIA 解压/重采样或 ONC 预处理安排在 GPU 验证阶段、两次训练之间或明确限速运行。

只有在线重采样被证明确实是瓶颈时，才离线生成版本化的 `16 kHz mono` recording cache。离线任务
按 recording 使用进程池，每个子进程单线程，采用临时文件后原子改名，并保存源文件哈希、目标
采样率、通道规则、处理版本和完成清单；任务可断点续作，不预生成 S0/S1/S2 的窗口。这样既能利用
多核 CPU，又不会把动态采样退化成体量巨大的固定切片数据集。

S1/S2 实现时额外记录每 epoch 实际访问的 recording/vessel 数量、重复率和数据等待时间。若 GPU
利用率已持续接近饱和，提高 worker 数不会带来收益，应保留较小配置，把 CPU 资源留给聚合评测或
后台数据审计。

## 5. 数据角色与现成来源

### 5.1 DeepShip：有标签下游训练与内部评测

本地已有 609 条 WAV、约 47.22 h。继续使用当前三种协议，主要报告
`recording_disjoint` 和 `vessel_name_disjoint`。不得通过生成更多重叠切片把片段数增长表述为
数据集扩充。

### 5.2 ONC Oceans 3.0：条件触发的无标签水声领域适配

ONC 提供连续水听器录音、站点、时间和设备元数据，但原始录音通常没有对齐的船型声学标签。
它适合多站点无标签自监督，不直接作为现成四分类集。项目默认不下载和训练大规模 ONC；只有
第 8.10 节的证据门满足时，才选择 10～20 h 多站点音频进行一次领域适配原型。

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

### 6.2 E2：连续四分类长上下文外测（未来工作）

E2 不属于当前项目完成条件。只有 10～20 s 长上下文已经在 DeepShip 严格协议下显示稳定价值，且
需要进一步验证这一结论时，才考虑构建连续四分类外测集。其目标是验证完整系统能否泛化到预训练
和微调阶段均未见的水域、水听器和物理船舶。

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
| B3 | 通用预训练 raw Conformer，20 s | 通用预训练迁移 | 预训练模型能否以现实成本迁移到水声分类 |
| B3-S | B3＋最佳 recording/vessel-balanced 动态采样 | 样本利用方式 | 去除重叠冗余和组规模偏置是否改善迁移 |
| B4（条件） | 通用 checkpoint → 10～20 h 多站点水声自监督 → 与 B3 相同的 DeepShip 微调 | 领域适配 | 已确认的语音—水声域偏移能否被小规模无标签适配缓解 |

B4 不是固定下一阶段，也不是在已经完成监督微调的 B3 权重上继续自监督。只有 B3 相对 B0/B1
已显示内部价值、但 PORTIA development 明显不足时，才从相同通用 checkpoint 重新开始
10～20 h 领域自监督，然后使用 B3 系列最终选定的相同 DeepShip 监督微调、采样和损失协议。
没有内部迁移价值时，不用更多无标签数据救活该路线。MIL、增强和双前端不进入当前最佳系统线。

大型 B2 scratch Conformer 已从必做矩阵删除。这不是声称 scratch 一定无效，而是主动限制研究
问题：619 M 参数从零训练的时间、数据和优化成本过高；即使完成，也不能单独消除 raw 前端、规模
和优化难度的混杂。若 B3 不能优于约 53 万参数的 B0，本项目把它解释为“当前通用预训练迁移没有
现实系统优势”，不继续扩大计算投入。

### 8.2 两条并行研究线与架构结论边界

B 系列用于比较完整系统，不能把预训练 raw Conformer 与 Mel CNN 的差值全部归因于 Transformer
架构。G 系列使用相同 log-Mel、相同上下文、训练数据和预算，比较纯 CNN、容量匹配对照和轻量
temporal self-attention。3 s 核心比较已完成；L20 复核仍保持三模型内部严格同配方。

因此两条线分别形成结论：

- B3/B3-S 对 B0/B1：通用语音大模型迁移是否值得其计算与适配成本；
- G1 对 G0/G0-C：全局时间注意力是否在专用水声频谱网络中有独立价值。

G 系列是 CNN/attention 混合架构，只支持“全局注意力”结论，不等价于纯 Transformer 端到端
因果比较。详细方案见 [`lightweight_global_attention_plan.md`](lightweight_global_attention_plan.md)。

### 8.3 上下文消融 T

当前 raw Conformer 已使用 20 s，不要求为了与 3 s CNN 表面一致而缩短输入。也不再默认运行
3/10/20/30 s 全网格：只有最佳 B3/B3-S 已接近或优于 CNN、并值得继续优化成本时，才补一个 10 s
效率对照；若 20 s 相对 10 s 的主指标提升不足 1 pp，则选择 10 s。30 s 默认删除。

G 系列 3 s 首轮显示 G1 不占优，但事后感受野审计发现 3 s 经 CNN 后只有约 50 个时间位置，局部
路径已接近覆盖全片段。因此只增加一次共同 20 s 复核，三模型均为约 316 个 CNN 后时间位置。
L20 内部的 G1−G0/G0-C 可以归因于结构；3 s 与 L20 同时改变了时长、采样和优化器，跨配方差值
不能单独归因于时长。10 s、30 s 和完整上下文网格删除。

### 8.4 微调策略消融 F

微调研究围绕“需要改动多少通用表征”展开。第一轮不叠加 LoRA、增强或双前端。

| ID | 策略 | 作用 |
|---|---|---|
| F0 | 冻结卷积前端和 24 层 Conformer，只训练 ASP/head | 判断通用表示能否直接浅层迁移 |
| F1a | 解冻最后 2 个 Conformer blocks | 保守任务适配 |
| F1b | 解冻最后 4 个 Conformer blocks | 单 seed 明显过拟合，且未观察到优于 F0 |
| F1c | 解冻最后 8 个 Conformer blocks | 检查更深任务适配是否必要 |
| F2 | 冻结卷积 feature encoder，微调全部 Conformer blocks | 调整全部高层时序表示但保留底层波形特征 |
| F3 | 卷积前端和 Conformer 全量微调 | 检查底层语音声学特征是否也需重构 |
| F4 | LoRA/Adapter 等参数高效微调 | 仅在 F1 过拟合或 F2/F3 显存受限时加入 |

已完成 `F0 → F1b`。当前证据不支持 F1c/last-8；先在 F0 上分离采样因素。只有最佳 S1/S2 通过
vessel 主指标幅度门后，才用 F1a/last-2 检查“轻量表征适配＋去冗余采样”是否优于纯冻结。F2/F3
只有在 F1a 形成稳定增益后保留。所有 backbone 参数使用远低于 pooling/head 的学习率，避免快速
破坏预训练表示。

### 8.5 组平衡动态采样消融 S

当前以 F0-S0 为对照。F0-S1 已完成；它保持相同 optimizer step 数、20 s 上下文、优化器、head
学习率和分类目标，并将 recording macro-F1 从 0.4181 提高到 0.5718，但 vessel macro-F1 只从
0.4726 提高到 0.4799。因为 S1 仍存在 24～1165 次的类内 vessel 暴露差异，当前运行唯一变化为
`vessel_balanced_dynamic` 的 F0-S2。

采用两级判定：

1. seed 42 探索中，只有 validation 主组级 macro-F1 提高至少 1 pp 且另一组级指标不退化的方案
   才进入候选；
2. 正式多 seed 中，只有平均组级指标改善且 paired bootstrap 大部分重采样差值为正，才替换 S0。

采样消融回答的是“通用预训练表示能否在减少相邻窗口冗余和组规模偏置后更好迁移”，不把动态
裁剪生成的新窗口描述为数据集扩充。

### 8.6 核心轻量全局注意力比较 G

G 系列从条件诊断升级为核心研究线，但首轮严格限制为三个模型：

| ID | 配置 | 唯一新增因素 | 回答的问题 |
|---|---|---|---|
| G0 | 当前 MA-CNN-A，共同上下文的 64-bin log-Mel | 无 | 统一训练配方下的强 CNN 基线 |
| G0-C | G0＋参数匹配的 temporal Conv/MLP block | 与 G1 近似的新增容量/FLOPs，不含 MHSA | 单纯增加容量是否足以解释收益 |
| G1 | G0＋1 层 gated temporal axial Conformer-lite block | 保留完整时频图并加入全局时间 self-attention | 全局交互是否提供独立组级收益 |

G1 在现有 `refine_time/refine_freq` 后、全局池化前保留完整 `[B,C,F',T']` 特征图，将其变形为
`[B×F',T',D]`，使一套共享 block 在每个频率位置沿完整时间轴做 attention；完成后恢复二维特征图，
再与原 CNN feature map 做近零初始化的门控残差。默认 `d_model=128`、4 heads、FFN expansion 2、
一个 block。注意力前不平均频率，最终复用 G0 的全局池化与分类器。预计总参数约 0.68～0.75 M，
实际实现后报告精确参数和 FLOPs。

3 s 首轮固定相同 vessel-name manifest、S0、SGD、总 step 和 validation vessel macro-F1 选模，
结果为 G0/G0-C/G1 vessel **0.5822/0.5855/0.5132**，G1 未通过。唯一 L20 复核固定相同 manifest、
20 s、S2 动态采样、AdamW、有效 batch 16 和总预算。只有 G1-L20 相对 G0-L20 至少 +1 pp、同时
优于 G0-C-L20 且 recording 最多下降 1 pp，才补多 seed；否则停止架构扩展，不运行 G1-L2、G2
或大型 AST/MGAE/UATR。

### 8.7 条件触发的小规模自监督原型 D

| ID | 无标签水声数据 | 目的 |
|---|---|---|
| D0 | 不做水声适配 | 通用预训练基线 |
| D1 | 10～20 h，多个站点 | 低成本判断水声领域适配是否值得继续 |

D1 只实现一种与通用 checkpoint 相容的 masked/contrastive 预训练目标，不并行比较多套 SSL 方法。
若 D1 没有在严格 DeepShip validation 或 PORTIA development 上带来稳定的 2～3 pp 改善，或使另
一项明显退化，则停止水声自监督。100～500 h 扩容只列为未来工作，不属于当前项目计划。

### 8.8 通用权重与水声适配权重的公平微调

B3 与 B4 必须使用相同的 DeepShip split、上下文、pooling/head、优化器、训练预算、模型选择规则
和微调深度。B4 直接复用 B3 已选定的配置，不重新搜索完整解冻深度，以免把额外调参预算误计为
领域自监督收益。若确有训练稳定性问题，最多补充 F0 与 F1a 两个低成本检查。B4 只提高 DeepShip
而不提高 PORTIA 时，不能声称领域自监督改善了泛化。

### 8.9 标签效率消融 L

在选定 CNN 和 Conformer 上使用 DeepShip 训练数据的 10%/25%/50%/100%。子集按 recording 或
vessel group 分层选取，不能随机抽三秒片段。该实验判断预训练 Conformer 是否能以更少标签达到
CNN 全量标签水平。

### 8.10 路线进入与停止条件

| 已观察到的证据 | 下一步 | 当时允许形成的中间结论 |
|---|---|---|
| 旧 F0 在第 14 epoch 显示明显过拟合 | 停止旧任务和 F1b 队列；切换组级选模与 step-level scheduler | 旧设置能拟合训练片段，但不能形成模型排名 |
| 新版 F1b−F0 为 −0.76 pp，paired CI 跨零且 F1b 明显过拟合 | 不运行 last-8；以 F0 为低方差基线进入 S1 | 当前单 seed 不支持更深监督微调 |
| F0-S1 vessel 仅 +0.73 pp、CI 跨零，但 recording +15.37 pp 且区间为正 | 因残余 vessel 暴露差异运行 F0-S2 | recording 平衡有效，但尚未证明未见船名泛化提升 |
| F0-S1 比 F0-S0 提高至少 1 pp，recording 指标不退化 | 保留 S1，随后检查 F1a/last-2 | 去除窗口冗余有利于通用表示迁移 |
| F0-S1 无提升且不存在明显残余 vessel 暴露偏置 | 停止 S 分支，进入上下文和预训练控制 | 采样冗余不是当前主要瓶颈 |
| S1/S2 比 S0 提高组级 validation，且另一组级指标不退化 | 将最佳动态采样用于后续 B3/B4 | 去除窗口冗余或组规模偏置有利于迁移 |
| B3/B3-S 不优于小型 CNN | 不训练大型 scratch、不启动 ONC；冻结最低成本 Conformer 结果 | 当前通用语音大模型迁移不具现实系统优势 |
| B3/B3-S 优于 CNN | 保留迁移路线，但仍运行核心 G0/G0-C/G1 | 证明迁移有效，不自动把收益归因于 Transformer 架构 |
| S2 结束且 B3 配置冻结 | 实现并顺序运行 G0、G0-C、G1 seed 42 | 单独检验全局时间注意力的价值 |
| G1 相对 G0 至少 +1 pp、优于 G0-C 且 recording 不退化 | 补 G0/G0-C/G1 多 seed 和 paired bootstrap | 全局注意力出现值得正式验证的正趋势 |
| G1 与 G0-C 提升相近 | 不进入 G2 | 收益更可能来自新增容量而非全局注意力 |
| G1 只提高 recording、不提高 vessel | 停止或只允许共同上下文诊断 | 全局模块利用录音规律，但未改善未见船名泛化 |
| G1 相对 G0/G0-C 没有稳定组级收益 | 停止架构扩展 | 当前输入、数据和协议不支持轻量全局注意力优势 |
| G1-3s 未通过且 3 s 局部感受野近全局 | 只运行共同 G0/G0-C/G1-L20 | 区分短输入限制与注意力设计无效 |
| G1-L20 仍不优于 L20 两个对照 | 停止 G 分支 | 当前严格协议不支持该轻量全局注意力设计 |
| B3/B3-S 内部接近/优于 CNN，但 PORTIA development 明显不足 | 启动一次 10～20 h 多站点 ONC 领域自监督 | 存在值得验证的语音—水声域偏移 |
| D1 没有稳定改善 2～3 pp，或另一项指标退化 | 停止水声自监督，不扩容 | 当前 SSL 目标或数据选择不成立 |
| D1 显示稳定正趋势 | 将其记录为原型结果；大规模扩容列入未来工作 | 水声领域适配有进一步研究价值 |

## 9. 指标、统计与成功标准

### 9.1 主要指标

- DeepShip `vessel_name_disjoint`：vessel-name-group macro-F1 和 Accuracy；
- DeepShip `recording_disjoint`：recording macro-F1 和 Accuracy；
- E1：MMSI-level macro-F1、balanced accuracy；
- E2（未来工作）：passage-level 与 MMSI-level macro-F1；
- 每类 recall、混淆矩阵、距离/站点分层结果；
- 参数量、训练显存、推理时延和每小时音频处理耗时。

### 9.2 Validation 聚合与模型选择规则

从下一版训练管线开始，每个 epoch 的 validation 不只计算 segment loss/accuracy，还保存窗口概率
并执行与测试阶段一致的 recording、vessel 等权聚合。模型选择和 early stopping 使用与研究问题
一致的组级指标：

| 协议 | 主选模指标 | 平分时依次比较 |
|---|---|---|
| `vessel_name_disjoint` | validation vessel macro-F1 | vessel Accuracy → recording macro-F1 → validation loss |
| `recording_disjoint` | validation recording macro-F1 | recording Accuracy → segment macro-F1 → validation loss |
| `segment_level` | validation segment macro-F1 | segment Accuracy → validation loss |

每个 checkpoint 写入选模层级、主指标、平分指标和对应聚合预测。validation/test 的动态采样必须
关闭，使用固定窗口集合；test 仍只在配置、checkpoint 和聚合规则冻结后运行一次，绝不参与 early
stopping。

已停止的旧 F0 使用 segment validation accuracy 和 epoch-level scheduler，仅保留为优化诊断。
新版 F0/F1b 已改用组级规则；F1b 的 train/validation 分叉和 bootstrap 均不支持 last-8。开发期
训练默认只写 `validation_complete`，不读取 test；只有最终配置冻结后才显式启用 test。

第二版调度以 optimizer update 为单位：F0/F1b 上限 30 epoch，每 epoch 约 1,750 次 update，前
5% update
从峰值学习率的 10% 线性 warmup，之后逐 update cosine decay 到 `1e-6`。F0 与 F1b 的 head 峰值
均先使用 `1e-4`；F1b encoder 峰值先使用 `5e-6`。early stopping patience 改为 5，并由上述组级
主指标触发。F0-S1/S2 均最多 8 epoch、patience 3，并要求主指标至少提高 0.005 才重置
早停；其余 AdamW、`weight_decay=1e-2`、BF16、有效 batch size 8、head 峰值 `1e-4` 和 gradient
clipping 1.0 保持不变。探索候选进入下一阶段仍要求相对 F0-S0 至少提高 1 pp。

### 9.3 正式运行和置信区间

- 探索阶段：固定 seed 42；
- 正式候选：建议 seed 42/43/44/45/46；
- 在相同测试 group 上做 paired bootstrap，报告模型差值的 95% 置信区间；
- 测试集选择、阈值和聚合规则在正式运行前冻结。

当前 vessel-name test 约 32 个 group，一个 group 约对应 3.125 个 Accuracy 百分点，因此小于
3 个百分点的单次变化不能视为稳定证据。

### 9.4 性能成功标准与项目完成标准

以下是正向性能成功标准，用于判断模型是否取得实质提升，不作为项目能否完成的必要条件：

1. DeepShip vessel-name-group macro-F1 比 B0 提高至少 5 个百分点，Accuracy 不明显退化；
2. DeepShip recording-level macro-F1 提高至少 3 个百分点，Accuracy 不明显退化；
3. E1 PORTIA MMSI-level macro-F1 提高 5 个百分点；
4. 四类中至少三类 recall 提高，不能仅由单一大类驱动；
5. paired bootstrap 的模型差值 95% 置信区间下界高于零，或在样本量不足时至少大部分
   bootstrap 重采样差值为正；
6. 低标签量下表现出比 CNN 更缓慢的性能下降。

即使上述性能标准未满足，只要完成以下证据链，项目仍视为完成并可以据实形成负结论：

1. B0/B1/B3 至少覆盖专用 CNN、长上下文和通用预训练迁移；G0/G0-C/G1 覆盖纯 CNN、容量对照和
   轻量全局注意力；
2. 所有模型使用冻结的 recording/vessel split，并按对应组级 validation 指标选模；
3. 报告多 seed 或配对 bootstrap、不确定性、参数量、显存、训练耗时和推理成本；
4. 完成 E1 PORTIA 外部评测，或明确记录数据/标签条件导致其不可执行的原因；
5. 能分别说明通用预训练迁移、轻量全局注意力、上下文和采样的贡献，并按第 8.10 节停止无收益
   分支；不要求用大型 scratch 完成完全因果的预训练归因；
6. 将正向或负向结论、失败模式和复现配置整理为最终实验报告。

## 10. 实验前预期与可得结论

下表是研究假设和工程目标，不是已经得到的结果，也不能在论文中写成实测值。

| 对比 | 合理预期 | 若成立可得结论 |
|---|---:|---|
| B1 vs B0 | 严格指标 +1～4 pp | 长上下文本身有效 |
| B3 vs B0/B1 | 方向未知 | 通用预训练迁移是否值得数百倍参数和训练成本 |
| B3-S vs B3 | recording/vessel +0～3 pp | 组平衡采样可能提高预训练表示的有效利用率 |
| B4 vs 匹配的 B3 系列对照 | 内部或外测 +2～3 pp | 小规模水声领域适配值得作为原型保留 |
| G0-C vs G0 | 方向未知 | 单纯增加约 15～20 万参数是否有效 |
| G1 vs G0/G0-C | vessel 至少 +1 pp 且 recording 不退化 | 频谱输入下的轻量全局注意力具有独立价值 |

结果解释规则：

- 只有 B1 提升：收益来自长上下文，不能声称 Conformer 架构更强。
- B3 明显优于 B0/B1，但 G1 约等于 G0/G0-C：大模型迁移有效，但不能声称轻量全局注意力本身
  有优势；收益可能来自预训练或模型规模。
- B3-S 提升：主要问题之一是重复窗口和组规模偏置，而不是模型容量不足。
- G1 明显优于 G0 和 G0-C，且 E1 同时改善：支持频谱注意力表示更可迁移。
- G1 与 G0-C 提升相近：收益主要来自容量，不能归因于注意力。
- G1 同时优于 G0/G0-C，但 raw Conformer 不优于 CNN：专用频谱全局注意力有效，瓶颈更可能是
  通用语音 raw 前端或预训练域匹配，而不是 attention 机制本身。
- G0/G0-C 不弱于 raw Conformer 与 G1：当前严格协议下局部频谱归纳偏置或额外容量更合适，
  不能声称全局注意力占优。
- B4 在 DeepShip 基本不变但外测提高：支持领域适配改善跨环境泛化。
- Segment accuracy 很高而 vessel/MMSI 不变：不能声称真实泛化改善。

保守目标区间为：vessel-name-group Accuracy 从当前约 53.13% 提升到 61%～68%，recording
Accuracy 从约 70.19% 提升到 76%～82%。该区间仅用于资源规划，不是承诺结果。

### 10.1 项目与简历叙事

推荐主叙事不是“用更大的 Conformer 替换 CNN”，而是：

> 现有水声船舶分类容易受到片段级数据关联影响。先建立 recording/vessel-disjoint 的可信评测，
> 再用两条研究线分别检验通用语音自监督 Conformer 的现实迁移价值，以及在专用频谱 CNN 中加入
> 轻量全局时间注意力能否改善未见船名泛化；最后在未见水域与 MMSI 的 PORTIA 上检验泛化。只有
> 已证明通用预训练有效且外测暴露明显域偏移时，才用小规模多站点无标签 ONC 水声继续自监督，
> 检验领域适配是否值得进一步研究。

核心贡献优先收敛为四项：

1. 严格的 recording/vessel 身份隔离和外部 MMSI 评测；
2. 通用语音自监督模型向水声船舶分类的迁移收益、成本或负结果证据；
3. 通过 G0/G0-C/G1 分离局部 CNN、额外容量和全局时间注意力贡献；
4. DeepShip 内部性能与 PORTIA 外部泛化之间的完整证据链。

已有船名 embedding 诊断用于说明身份捷径风险。身份不变 objective 不与首轮架构实验同时加入；
若未来启动，只能在最佳架构冻结后单独消融。G0/G0-C/G1 是核心且有明确停止门的轻量研究线，
不扩展为大量专用频谱网络复现。

只有小规模 ONC 原型实际改善严格 validation 或外测 development 时，才增加第五项探索性贡献：
多站点无标签水声领域适配。

### 10.2 可能形成的最终结论

**最强结论**：通用预训练 Conformer 在严格 DeepShip 协议和 PORTIA MMSI 外测上均优于 CNN；
组平衡采样和长上下文进一步提高未见船舶指标。若条件触发的 ONC 原型也改善跨水域泛化，则可以
形成“通用预训练 → 任务微调 → 小规模领域适配”的层级迁移故事。

**中等结论**：Conformer 在 DeepShip 上优于 CNN，但直接跨域无优势；ONC 领域适配后 PORTIA
才改善。此时结论是通用预训练提供基础表示，但水声领域自监督是获得外部泛化的必要环节。

**迁移有效但轻量注意力不占优**：B3 明显优于 CNN，但 G1 不优于 G0/G0-C。此时应把重点放在
预训练迁移或模型规模，而不是声称全局注意力本身更强。

**迁移失败但轻量注意力有效**：raw Conformer 未优于 CNN，但 G1 稳定优于 G0/G0-C。此时结论是
全局依赖建模对水声有价值，主要瓶颈更可能是通用语音预训练、raw 前端或迁移成本，而不是
attention 机制本身。

**长上下文故事**：10～20 s 在严格 group 指标上稳定优于 3 s，但架构差异有限。此时结论是船舶
中长时动态具有判别价值，系统收益主要来自上下文和聚合策略。

**负结果但仍成立的项目结论**：严格隔离和外测下，大型语音预训练 Conformer 未稳定优于 CNN，
或 ONC 自监督无法修复域偏移。此时可以据实说明模型规模和端到端表示不天然带来水声泛化，局部
频谱结构、预训练领域匹配或数据身份隔离比增加模型容量更重要。若此时 G1 同时优于 G0 和 G0-C，
则问题主要是通用语音 raw 前端/预训练匹配；若 G1 也不优于两个对照，则当前数据与协议没有支持
轻量全局注意力优势。

## 11. 下载量、磁盘和数据获取顺序

### 11.1 大小估算

16 kHz、单通道 PCM16 约为 115 MB/h。ONC 原始数据若为 64～96 kHz、24-bit、单通道，约为
0.69～1.04 GB/h。实际压缩率和站点产品格式会改变下载量。

| 阶段 | 数据 | 预计下载量 | 16 kHz PCM16 处理后 |
|---|---|---:|---:|
| 代码原型 | 现有 DeepShip＋预训练权重 | 所选 large checkpoint 约 2.5 GB | 无新增外部音频 |
| 条件 SSL 原型 | ONC 10～20 h | 约 7～20 GB | 约 1～2.3 GB |

PORTIA v2 完整压缩音频约 75 GB；官方建议在同时保留压缩包和解压 WAV 时准备至少约 150 GB。
若再保存 16 kHz 单通道副本，处理后约 9 GB。批量处理应尽量边解压、边选通道和重采样，避免
长期保留多份原始副本。

除已开始下载的 PORTIA 外，ONC 第一轮新增下载量上限为 10～20 GB。加入完整 PORTIA、处理缓存、
模型权重和多个 checkpoint 后，建议外置盘至少保留 500 GB 可用空间。当前项目不为 100～500 h
ONC 扩容预留下载或训练任务。

### 11.2 按证据扩充数据

当前 PORTIA 已开始完整下载；这不意味着提前使用外测调参。后续数据扩充按研究证据决定：

1. 用现有 DeepShip 和通用预训练权重完成 B3 的冻结、有限解冻和 S0/S1/S2 采样比较；不训练大型
   scratch Conformer。
2. 在相同冻结 DeepShip 协议上完成 G0/G0-C/G1，先回答轻量全局注意力问题，不需要新增数据。
3. PORTIA 下载与标注审计可并行完成，但封存 test 在 B3/B4/G 候选及聚合规则全部固定前保持
   不可见。
4. 先用冻结 B3 和 G 系列候选运行 PORTIA development，诊断内部提升是否受外部领域偏移限制。
5. 只有“B3 内部接近或优于 CNN、PORTIA development 显示外部明显不足”时，下载 ONC 10～20 h
   多站点音频并运行 SSL 原型。
6. 小规模 SSL 若没有稳定的 2～3 pp 改善或使另一项指标退化，立即停止；若有效，只记录为原型
   证据，把 100～500 h 扩容列入未来工作。
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
- 为 E1 定义独立版本和 manifest schema；E2 schema 仅在未来实际启动时再冻结。

验收：仅凭配置和本文可明确每个实验改变了哪个变量。

### 阶段 1：DeepShip raw Conformer 基线

- 已完成 optimizer-step scheduler、组级选模、resume 和动态终端日志测试；
- 已完成新版 F0，并在 F1b 的过拟合趋势足够明确后主动停止；不恢复 F1b，不运行 last-8；
- 已用 paired bootstrap 确认单 seed 下 F1b 没有可辨认收益，以 F0 作为下一阶段低方差基线；
- 开发期训练默认封存 test，仅保存 validation 最佳 checkpoint 和组级预测；
- 已完成 S1；它改善 recording 但未通过 vessel 幅度门，因此暂不进入 F1a。

验收：现有三个 DeepShip 协议均无 group 泄漏；validation/test 使用相同聚合定义；模型选择不再
依赖 segment accuracy；训练、resume、推理和指标可复现。

### 阶段 2A：完成组平衡采样并冻结迁移线

- 已完成 S1 的确定性 recording-balanced 动态裁剪、暴露报告和 CPU 等待统计；
- S1 显著改善 recording 指标，但 vessel 主指标仅增加 0.73 pp；
- 已完成 S2；其 vessel macro-F1 为 0.5625，按预设点估计门通过，但 paired bootstrap 区间跨零；
- 已完成唯一一次 F1a；它不改变 S2 数据分布，只检查解冻最后 2 个 block 的边际价值；
- F1a vessel macro-F1 为 0.6435、recording 为 0.4380；相对 F0-S2 vessel +8.10 pp，但 paired
  区间跨零。它作为单 seed 迁移候选保留，不继续扩大解冻，也不读取 test；
- 全部选择只使用 validation recording/vessel 指标，不根据 DeepShip test 反复调参；
- 不运行大型 scratch；若最佳 raw Conformer 不优于小型 CNN，直接记录迁移成本/收益负结论；
- 冻结 B3 系列最终 checkpoint、采样、损失和聚合规则。

验收：能够说明微调深度、动态采样和上下文对通用预训练迁移的影响；B3 系列每个保留因素均有
独立的组级 validation 证据，并明确计算成本和未完成 scratch 所限制的结论边界。

### 阶段 2B：核心轻量全局注意力比较

- 已实现 G0、G0-C 和 G1，不修改正在运行的 Conformer checkout；
- G1 保留完整时频图并使用 1 层共享 temporal axial Conformer-lite；G0-C 使用相同投影、频率位置
  信息和门控融合，仅以局部/扩张时间卷积替换 MHSA，并匹配新增参数和计算量；
- 3 s 三组已完成：G1 的 vessel/recording 均低于 G0/G0-C；
- 已冻结唯一 L20 复核：三组共同使用 20 s log-Mel、S2、AdamW、有效 batch 16、相同总预算和
  validation vessel macro-F1 选模；短录音的 padded Mel/CNN 时间位置统一 mask；
- 3 s 已通过数值回归、窄频带、梯度、resume、参数/FLOPs 和无 test 读取测试；L20 新增动态
  log-Mel、padding mask、梯度累积与优化器测试，服务器真实数据 smoke 后顺序运行 seed 42；
- L20 seed42 的 G0/G0-C/G1 vessel macro-F1 为 0.6261/0.6440/0.6741，但 G1 recording
  由 G0 的 0.6187 降至 0.5604，且单 split paired bootstrap 区间跨零；
- 为区分训练随机性与 vessel split 随机性，冻结 3 个 split seed × 3 个 model seed × 3 个模型的
  DeepShip-only 全交叉矩阵，复用现有 split42/seed42 三项并顺序新增 24 项；只用 validation；
- 最终用 split→model seed 分层 bootstrap、逐 split 均值和 recording 代价共同判定；
- 若 L20 不通过即停止 G 分支，不再运行两层 attention、双轴 attention 或更多上下文点。

验收：能够区分额外容量与全局时间注意力；所有比较使用相同输入、训练和选模协议；结论用词限定
为轻量全局注意力，不泛化为所有纯 Transformer。详细验收见
[`lightweight_global_attention_plan.md`](lightweight_global_attention_plan.md)。

### 阶段 3：外部开发集诊断与条件触发的小规模水声自监督

- 审计 PORTIA 音频并只运行 development，判断是否存在值得研究的领域偏移；封存 test 不运行；
- 仅当 B3 内部接近或优于 CNN、但 PORTIA development 显示明显外部不足时进入 ONC 自监督；
- 下载并审计 10～20 h 多站点 ONC；
- 只实现一种与通用 checkpoint 相容的 masked/contrastive 目标；
- 检查表示方差、协方差、下游 probe、训练稳定性及 B4 相对 B3 的外测 development 变化。

验收：外部诊断不使用封存 test；无坍缩；B4 至少在严格 validation 或外测 development 上稳定
改善约 2～3 pp，且另一项不明显退化，否则停止水声自监督。有效时只记录为原型结果，不自动扩容。

### 阶段 4：最终评测和报告

- 冻结代码 commit、配置、checkpoint 选择规则和聚合规则；
- 运行 DeepShip 正式多 seed 和 E1；E2 只作为未来工作；
- 生成 paired bootstrap、混淆矩阵、距离/站点分层和资源报告；
- 对照第 9.4 节性能标准和完成标准给出支持或否定结论。

验收：所有结论均能追溯到唯一数据 manifest、代码 commit、配置、模型权重和输出目录。

## 13. 主要风险与停止条件

| 风险 | 处理方式/停止条件 |
|---|---|
| Segment validation accuracy 选择到依赖局部片段的 checkpoint | 改用协议对应的 validation recording/vessel macro-F1；旧 F0/F1b 仅作探索 |
| 动态采样减少冗余但覆盖不足 | 固定 optimizer step 数并记录每 epoch 的 recording/vessel 覆盖；组级指标不提升则回退 S0 |
| 大型 scratch Conformer 成本高且不能消除多种混杂 | 不运行；将问题限定为通用预训练迁移的现实系统价值 |
| ONC 与 DeepShip 站点相近造成隐性域重合 | 多站点训练；E2 站点完全留出 |
| Oceanship 弱标签或连续性不足 | 不作为 E2；改用留出 ONC＋AIS 自建 |
| PORTIA 只有 3 s | 仅用于 E1；不据此评价长上下文收益 |
| AIS 最近船不是主要声源 | 多船/距离筛选和人工审核；报告弱标签限制 |
| 轻量注意力提升实际来自额外参数 | 增加 G0-C 容量对照；G1 必须同时优于 G0 和 G0-C |
| G 系列无限扩展使项目偏离研究问题 | 首轮只运行 G0/G0-C/G1；不通过 +1 pp 门即停止，最多触发一个后续扩展 |
| 20 s 相比 10 s 无收益 | 选择 10 s，停止扩大上下文 |
| 10～20 h 多站点 SSL 无稳定收益 | 停止水声自监督；不下载或训练 100～500 h 数据 |
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
10. [MGAE-Net：多尺度频谱 Transformer 与 MMSI-disjoint 评测](https://doi.org/10.1016/j.oceaneng.2026.126226)
11. [MBAT：individual-vessel 泛化与船舶身份不变训练](https://doi.org/10.1121/10.0036456)
12. [项目现有隔离训练路线](strict_isolation_training_plan.md)
13. [轻量全局注意力架构实验计划](lightweight_global_attention_plan.md)
14. [Attention Augmented Convolutional Networks](https://arxiv.org/abs/1904.09925)
15. [Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778)
16. [SSAST](https://arxiv.org/abs/2110.09784)
17. [项目存储布局](../guides/storage_layout.md)
