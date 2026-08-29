# DeepShip 轻量全局注意力架构实验计划

最后更新：2026-08-29  
状态：3 s 与 20 s 的 G0/G0-C/G1 seed 42 均已完成。L20 的 G1 在 validation vessel macro-F1
上出现正趋势，但 recording 指标下降且单 split bootstrap 区间跨零，不能形成稳定结论。当前执行
冻结的 `3 个 vessel split × 3 个训练 seed × 3 个模型` DeepShip-only 重复矩阵；现有
split42/seed42 三项复用，新增 24 项 validation-only 正式实验。

## 1. 研究问题与边界

本实验线回答：

> 在 DeepShip 未见船名隔离条件下，保留领域专用 log-Mel CNN 的局部时频归纳偏置，同时加入
> 少量全局时间自注意力，能否比纯 CNN 获得更好的 vessel-level 泛化？

它与 raw Conformer 迁移线并行，而不是 Conformer 失败后的临时补救：

| 研究线 | 核心问题 | 允许形成的结论 |
|---|---|---|
| P：预训练迁移 | 大型通用语音 Conformer 能否以可接受的适配成本迁移到水声分类 | 通用预训练的迁移价值、成本和失败风险 |
| G：轻量全局注意力 | 在相同专用频谱输入和训练预算下，全局注意力是否比局部 CNN 提供额外价值 | 全局依赖建模是否有独立贡献 |

G 系列是 CNN/attention 混合架构，不等同于纯 Transformer，也不是严格的原始波形端到端系统。
因此本项目不根据 G 系列结果笼统声称“所有 Transformer 都适合/不适合水声”，而把结论限制为
“轻量全局时间自注意力在当前输入、数据规模和隔离协议下是否有效”。

大型 Conformer scratch 不再作为必做实验。若数亿参数的通用预训练模型在合理微调、采样和选模后
仍不能优于约 53 万参数的专用 CNN，项目可直接形成“跨域迁移的成本和失败率高于专用小模型”的
系统级结论；不额外消耗大量 GPU 时间追求纯架构因果分解。

## 2. 基础网络与新增信息路径

当前 MA-CNN-A 约 532,166 个参数，使用 16 kHz、64-bin log-Mel。三条非对称卷积分支分别
使用 8/16/32 的时间和频率卷积核；现有 `PaperAttentionFusion` 根据全局池化结果重标定通道，
但不允许相距较远的时间位置直接交互。因此它是 channel attention，不是时间全局 self-attention。

首选 G1 结构：

```text
log-Mel [B, 1, F, T]
→ 现有三分支非对称 CNN
→ 现有 channel-attention fusion
→ refine_time + refine_freq
→ feature map H [B, 98, F', T']
→ pointwise projection 98 → 128
→ add frequency-coordinate embedding
→ reshape [B, 128, F', T'] → [B×F', T', 128]
→ depthwise Conv1d temporal positional encoding, kernel 9
→ 1 × shared Pre-LN temporal axial Conformer-lite block
   ├─ 4-head MHSA over the complete T' axis
   ├─ depthwise temporal Conv1d, kernel 15
   └─ FFN 128 → 256 → 128
→ reshape back to A [B, 128, F', T']
→ H_out = H + sigmoid(g) × Conv1x1(A, 128 → 98)
→ original AdaptiveAvgPool2d + Linear(98, 4)
```

实现约束：

- `d_model=128`、`num_heads=4`、每个 head 32 维；
- 首轮只放 1 个 block，FFN expansion 固定为 2；
- activation 使用 GELU，attention/FFN dropout 均为 0.1；
- stochastic depth 首轮为 0 或不超过 0.05；
- 门控参数 `g` 初始化为 `-2`，使训练开始时接近原始 CNN；
- 同一套 temporal block 在全部 `F'` 个频率位置共享参数，参数量不随频率位置数量增长；
- 使用归一化 Mel 频率坐标的轻量 embedding 保留绝对频率含义；G0-C 使用相同 embedding；
- 时间 padding mask 在 reshape 后复制到每个频率位置，直到 attention 完成；
- 注意力前不在时间或频率维做全局平均；只在 contextualized feature map 与原特征做门控残差后，
  复用 G0 原有的最终全局池化和分类器。

实际实现结果为：G0 532,166 参数，G0-C 679,593 参数，G1 694,057 参数；G0-C/G1 的新增参数
分别为 147,427/161,891，相对差 8.93%。在 `[1,1,64,94]` 输入上用 PyTorch flop counter 得到
G0/G0-C/G1 前向 FLOPs 约为 1.767/2.274/2.095 G（对应近似 MACs 为其一半），G0-C 与 G1
相差 8.53%。参数与计算量均达到预设 10%/15% 匹配门。CPU 延迟不作为最终部署结论；服务器
GPU 峰值显存和推理时延仍由正式审计补充。3 s 输入产生 94 个 Mel 帧、CNN 后约 50 个时间位置；
20 s 输入产生 626 个 Mel 帧、CNN 后约 316 个时间位置。L20 不改变三种模型之间的结构差异，但让
G1 的全局路径明显超过 G0-C 的有限局部感受野。L20 本地 CPU 前向审计得到 G0/G0-C/G1 约
11.34/14.54/13.41 GFLOPs；G0-C 与 G1 的 FLOPs 差 8.43%，参数与计算匹配门在长输入下仍满足。

首轮不使用二维全局注意力，也不在注意力前平均频率。3 s 特征图即可能产生约千级时频 token；
直接对 `F'×T'` 做二次复杂度注意力既增加成本，也把频率和时间当成同质维度。时间轴向注意力的
关系计算量约为 `F'×T'^2`，而二维全局注意力约为 `(F'×T')^2`。以 `F'=32、T'=48` 为例，前者
约 7.4 万个位置关系，后者约 236 万个。这样既保留频率特异的时间变化，也允许每个频带在完整
观察时长内建立全局依赖。

## 3. 受控实验矩阵

### 3.1 核心三组

| ID | 网络 | 相对 G0 的唯一变化 | 用途 |
|---|---|---|---|
| G0 | 原始 MA-CNN-A | 无 | 统一训练配方下重建强 CNN 基线 |
| G0-C | MA-CNN-A＋参数匹配的 temporal Conv/MLP block | 增加与 G1 近似的参数和计算量，但无全局 MHSA | 排除单纯增加容量的解释 |
| G1 | MA-CNN-A＋1 个上述 temporal axial Conformer-lite block | 保留完整时频图并增加全局时间交互 | 检验全局注意力的独立贡献 |

G0-C 使用与 G1 完全相同的 pointwise projection、频率坐标 embedding、位置卷积、FFN、投影回写
和门控残差，但用多尺度 depthwise/dilated temporal Conv1d 替代 MHSA，并调整 hidden width，
使新增参数与 G1 相差不超过 10%、MACs/FLOPs 相差不超过 15%。如果两项无法同时匹配，优先匹配
参数量，并在报告中列出计算量差异。这样 G1 与 G0-C 的主要结构差异才是全局内容相关交互。

G0 必须在新管线下重跑。不能直接把历史 CNN 数字作为唯一对照，因为历史 run 的 checkpoint 选择、
seed 数和新增 G 系列训练配方可能不同。

### 3.2 唯一上下文复核：L20

3 s 的 seed 42 结果已经完成：G0/G0-C/G1 的 validation vessel macro-F1 分别为
**0.5822/0.5855/0.5132**，recording macro-F1 分别为 **0.5694/0.6127/0.4910**。G1 相对 G0
下降 6.90 pp、相对 G0-C 下降 7.23 pp，均未通过保留门。

但该负结果存在一个事后发现的结构性限制：3 s 输入经 CNN 后仅约 50 个时间 token，而原卷积分支
与 G0-C 的局部卷积感受野已经覆盖片段的大部分甚至全部时间范围。因此 3 s 比较能说明当前 G1
实现不优于局部网络，却不能充分区分“全局依赖无用”和“输入根本没有额外长程关系可建模”。

为回答这一剩余问题，只增加一组共同 L20 复核：G0-L20、G0-C-L20、G1-L20 全部使用相同 20 s
输入、S2 vessel-balanced dynamic 采样、AdamW、batch/step 预算与 validation 选模。除此之外不再
运行 10 s 中间点、G1-L2、G2、head/hidden-size 搜索或二维全局注意力。L20 若仍不通过保留门，
G 分支停止；若通过，才建议补 seed 43/44，不能直接宣称确定性提升。

L20 seed 42 的 validation vessel macro-F1 为 G0/G0-C/G1=
**0.6261/0.6440/0.6741**；G1 相对 G0/G0-C 分别为 +4.80/+3.01 pp。对应 recording
macro-F1 为 **0.6187/0.5798/0.5604**，G1 相对 G0 下降 5.83 pp，因此没有通过原始双指标门。
同 vessel 配对 bootstrap 的 G1−G0 vessel 区间为约 [−10.12,+20.30] pp，`P(Δ>0)=0.747`；
这只能说明存在值得复核的趋势，不能证明注意力有效。

## 4. 数据、输入和公平比较

已完成的 3 s G0/G0-C/G1 固定：

- `vessel_name_disjoint` 冻结 manifest 和相同 manifest hash；
- 16 kHz、3 s、64-bin log-Mel，以及完全相同的归一化和窗口；
- 相同 train/validation 样本、batch 顺序规则、增强、loss 和总 optimizer steps；
- validation 使用固定窗口；test 保持封存；
- 模型选择使用 validation vessel macro-F1，平分时比较 vessel accuracy、recording macro-F1 和
  validation loss；
- 三个模型均保存 segment、recording、vessel 预测和混淆矩阵。

L20 复核固定：

- 仍使用同一个 `vessel_name_disjoint` 冻结 manifest；train/validation/test 的船名与录音隔离不变；
- train 使用 class→vessel→recording-balanced 动态裁剪，每 epoch 14,000 个 20 s 窗口；
- validation 使用冻结 3 s anchor 的中心扩展 20 s，所有模型看到完全相同的固定窗口；
- 短于 20 s 的录音在波形右侧补零，补零后的 Mel 帧清零，并在 CNN 后对 G0 池化、G0-C 局部模块
  和 G1 attention 使用同一个时间 mask；
- 16 kHz、64-bin log-Mel、AdamW `3e-4`、weight decay `1e-2`、warmup 5＋cosine、gradient clip 1；
- 物理 batch 4、梯度累积 4、有效 batch 16、BF16、最多 50 epoch、patience 8；
- validation vessel macro-F1 选模；test 继续封存。

L20 三组内部的差值可以归因于网络结构，因为数据、输入、采样、优化器和预算全部相同。3 s 与
L20 之间同时改变了时长、S0→S2 和优化器，因此跨配方差值只能称为“系统配方比较”，不能把
G0-L20 与 G0-3s 的差值单独解释为时长收益。若未来需要纯时长因果结论，必须另加相同 S2/AdamW
的 G0-3s 对照；当前项目不自动扩展该实验。

## 5. 训练与工程计划

当前实现入口：

- 模型：`src/models/ma_cnn_a.py`；
- validation-only 管线：`src/pipelines/mel_ml/train_deepship_macnna_global.py`；
- CLI：`scripts/train/train_deepship_macnna_global.py`；
- 统一 runner：`scripts/train/run_macnna_global_seed42.sh`；
- L20 runner：`scripts/train/run_macnna_global_l20_seed42.sh`；
- 重复 split 构建：`scripts/prepare/build_deepship_repeat_splits.py`；
- 重复单项/矩阵 runner：`scripts/train/run_macnna_global_l20_repeat.sh` 与
  `scripts/train/run_macnna_global_l20_repeat_matrix.sh`；
- 重复结果汇总：`scripts/eval/summarize_macnna_global_l20_repeats.py`；
- 参数/FLOPs 审计：`scripts/eval/audit_macnna_variants.py`；
- 冻结配置：`configs/experiments/macnna_global_v1.json` 与
  `configs/experiments/macnna_global_l20_v1.json`。

### 5.1 实施前基准冻结

1. 记录当前 `MACNNAClassifier` 的参数量、输入输出 shape 和 3 s MACs/FLOPs。
2. 固定新的 G 系列实验配置、seed、manifest hash、指标和输出 schema。
3. 把历史 MA-CNN-A 训练配方作为起点，但 G0/G0-C/G1 必须使用同一个优化器和 scheduler。
4. 若 attention 使用 AdamW 才稳定，则三组全部改用 AdamW 重跑；不能只给 G1 更有利的优化器。

3 s 首轮使用冻结的 SGD＋momentum。L20 因输入与训练动力学均改变，在运行结果前统一冻结为
AdamW；三模型不得使用不同优化器或 scheduler。优化器决定不读取 test，也不根据 G1 单独修改。

### 5.2 代码实施顺序

1. 将 MA-CNN-A 的卷积特征提取和分类头拆成可复用接口，同时保证原 `MACNNAClassifier` 数值行为
   不变。
2. 实现 pointwise projection、频率坐标 embedding、`[B×F',T',D]` reshape、位置 depthwise
   Conv1d、共享 Pre-LN temporal axial block 和 feature-map 门控残差。
3. 实现 G0-C 容量对照和参数/FLOPs 审计。
4. 扩展训练配置：`model_variant`、attention 超参数和 gate 初始化全部写入 run config。
5. 复用现有三级聚合评测；补齐 validation vessel macro-F1 选模、预测文件和 `test_evaluated=false`。
6. 增加动态终端进度、环境记录和退出完整性检查，但不修改正在运行的 Conformer checkout。

### 5.3 测试门

进入正式训练前必须通过：

- G0 重构前后在相同权重和输入下 logits 一致；
- G0/G0-C/G1 的 forward/backward、梯度有限性和 checkpoint resume 测试；
- batch size 1、不同 batch size 和至少两个合法时间长度的 shape 测试；
- gate 初始化时 G1 输出与 G0 接近，但 attention 参数仍能获得有限非零梯度；
- mask 在 `B×F'` 展开后与每个频率位置正确对应，padding 不参与 MHSA；
- 构造只在单个窄频带变化的输入，确认注意力前不存在隐式频率平均；
- 参数量匹配审计和无 test 读取测试；
- 1～2 batch overfit、短 smoke 和单 epoch throughput 测试。

### 5.4 正式运行顺序

```text
3 s：G0 → G0-C → G1（已完成，G1 未通过门）
→ L20 工程测试与参数/FLOPs、显存 smoke
→ G0-L20 seed 42
→ G0-C-L20 seed 42
→ G1-L20 seed 42
→ validation vessel/recording paired bootstrap 与容量、成本比较
→ seed42 出现 vessel 正趋势但 recording 下降、区间跨零
→ 完成 split/model seed 42/43/44 的 3×3 全交叉重复；每个 cell 均运行 G0/G0-C/G1
→ 分层 bootstrap 先重采样 vessel split，再在 split 内重采样训练 seed
→ 根据均值、split 一致性、区间和 recording 代价判定；不进入 G1-L2/G2/更多上下文搜索
```

L20 三个正式实验顺序运行，避免 GPU 并发使吞吐和峰值显存失去可比性。训练上限、early stopping、
物理 batch 和累积步数已经在结果产生前冻结，不能看到 G1-L20 后单独放宽。

## 6. 指标、统计与决策门

主指标：validation vessel macro-F1。次指标：vessel accuracy、recording macro-F1、recording
accuracy。辅助指标：segment macro-F1/accuracy、每类 recall、ECE 或 Brier score、参数量、FLOPs、
显存、吞吐和推理时延。

探索保留门：

1. `G1 − G0` 的 vessel macro-F1 至少为 +1 pp；
2. G1 同时优于 G0-C，避免把容量收益误认为注意力收益；
3. recording macro-F1 不低于 G0 超过 1 pp；
4. 提升不能只由一个类别 recall 驱动；
5. 无训练不稳定、非有限 loss 或明显校准恶化。

正式重复矩阵固定 split seed 42/43/44 与 model seed 42/43/44，三种模型完全配对。先在每个
split 内计算三训练 seed 的平均差，再对 split→seed 两层做 hierarchical bootstrap。支持“全局
注意力有帮助”至少要求：G1−G0 vessel 平均不少于 +1 pp、G1 平均高于 G0-C、三个 split 至少
两个平均为正、recording 相对 G0 平均下降不超过 1 pp，并报告区间和 `P(Δ>0)`。若 vessel 改善但
recording 持续下降，则结论是存在聚合层级权衡，而不是无条件有效。

三个 split 的 Cargo/Passenger/Tank validation 组构成明显变化；由于 Tug 仅有 16 个船名组且
当前固定容量预算存在唯一可行的最小容量组合，三个 split 的 6 个 Tug validation 组相同。因此
跨 split 结论主要降低前三类的划分随机性；Tug 的不确定性仍由训练 seed、逐类结果和后续独立
MMSI 外测体现，不能声称已完全消除 Tug 数据不足。

## 7. 结果解释矩阵

| 观察结果 | 结论 |
|---|---|
| G1 > G0 且 G1 > G0-C，vessel/recording 同时改善 | 轻量全局时间注意力具有独立价值 |
| G1 与 G0-C 都以相近幅度优于 G0 | 收益主要来自容量或额外非线性，不能归因于注意力 |
| G1 只改善 recording，不改善 vessel | 全局模块利用了录音级长时一致性，但未解决未见船名泛化 |
| G1-3s 无效、共同 L20 后有效 | 注意力价值依赖足够长的观察上下文；需多 seed 确认 |
| G1-3s 与 G1-L20 均不优于各自 G0/G0-C | 当前 DeepShip 严格协议不支持该轻量全局注意力设计，停止扩展 |
| G1 不优于 G0/G0-C，raw Conformer 也不优于 CNN | 当前数据规模与协议更支持局部频谱归纳偏置；不能外推到所有 Transformer |
| G1 优于 G0/G0-C，但 raw Conformer 不优于 CNN | 全局建模有效，主要瓶颈更可能是语音预训练/原始波形域匹配，而非注意力机制本身 |
| raw Conformer 优于 CNN，但 G1≈G0/G0-C | 大模型收益更可能来自通用预训练或规模，不支持轻量注意力的架构归因 |

身份不变损失、MBAT 式 objective、数据增强和动态采样均不与 G0/G0-C/G1 首轮同时加入。它们改变
训练目标或数据分布，应在最佳架构冻结后作为独立实验；否则无法判断收益来自结构还是正则化。

## 8. 与项目主线的衔接

1. 先完成当前 F0-S2，仅用 validation 判断 S0/S1/S2 的采样结论。
2. 冻结最低成本且表现最好的 raw Conformer 配置；不训练大型 scratch Conformer。
3. 无论 raw Conformer 最终是否超过 CNN，都执行核心 G0/G0-C/G1，因为它回答不同研究问题且
   训练成本远低于 619 M 参数模型。
4. 在 DeepShip 上冻结两条线的候选模型后，再进入 PORTIA development；封存 test 只运行一次。
5. ONC 10～20 h 自监督只有在通用预训练已显示内部价值、但 PORTIA development 暴露明显域偏移
   时才启动。没有内部迁移收益时，不用更多无标签数据救活该路线。

### 8.1 有船舶身份标注数据对注意力结论的作用

- PORTIA 固定 3 s，只验证短窗跨域分类，不验证 L20 长程注意力；
- Oceanship-FG 有大量 MMSI，但公开样本约 4～5 s；它适合在增加独立船数后复核短上下文模型，
  若要复核 L20 必须按 [`oceanship_onc_reconstruction_plan.md`](oceanship_onc_reconstruction_plan.md)
  的分阶段准入门回溯 ONC 连续录音；
- Belgian AIS 公开数据为 10 s 非重叠窗口，但匿名 `mmsi` 不能作为跨 AIS 文件稳定的船舶身份，
  且公开片段通常不相邻，因此不构造伪 L20 或 vessel-disjoint split。它按照
  [`belgian_public_attention_plan.md`](belgian_public_attention_plan.md) 只做四分类 G0/G1 的
  date-disjoint 跨水域复现；已申请的五天连续数据另行审计真实身份与连续性；
- DeuteroNoise 有连续长录音和人工单船筛选，但目标四类不均衡，只作为辅助长上下文证据。

外部数据不能代替 DeepShip 内部 G0/G0-C/G1 的架构归因，因为水域、设备、标签质量和船型分布
同时变化。先完成 DeepShip-only 重复；只有内部方向稳定后，才冻结模型并在一个合格外部数据集
上做一次独立复现。

## 9. 参考结构依据

- [Conformer](https://arxiv.org/abs/2005.08100)：卷积负责局部模式、注意力负责全局依赖的混合设计。
- [Attention Augmented Convolutional Networks](https://arxiv.org/abs/1904.09925)：在卷积中加入全局
  self-attention，并用受控结构比较局部与全局建模。
- [Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778)：谱图 token 上的纯注意力音频分类。
- [SSAST](https://arxiv.org/abs/2110.09784)：说明音频 Transformer 对数据规模/预训练较敏感，支持
  本项目优先保留 CNN 归纳偏置并只加入轻量注意力。
