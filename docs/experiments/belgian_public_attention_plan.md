# Belgian AIS 四分类全局注意力复现实验

状态：归档已完成大小/MD5 校验和解压；真实音频审计发现文件同时含单、双声道且少数时长不精确。
因此在正式训练前将协议重新冻结为“只接纳精确 10.000 s 文件；单/双声道均固定读取 channel 0”。

本实验使用公开的 Belgian AIS 标注 10 s 水声片段，在第二个水域中比较原始 CNN（G0）与加入
shared temporal axial self-attention 的 CNN（G1）。它回答的是：**在相同数据、采样和优化预算下，
给原始频谱 CNN 增加全局时间注意力，是否能稳定改善跨日期的四类船噪分类**。

本实验不把公开元数据中的匿名 `mmsi` 当作全局船舶身份。该 ID 在不同 AIS 输入文件之间可能
重复，因此不能声称得到真正的 vessel-disjoint 结果，也不能把指标命名为 vessel F1。申请中的
五天连续数据仍用于未来构造身份可核验、连续性更强的补充实验。

## 1. 数据准入与结论边界

公开数据集由 Gardencity 和 Grafton 两个站点、116 天采集，官方说明包含 27,524 个 10 s、单声道
48 kHz 片段，并提供 train/validation/test 文件列表。官方划分保证同一站点的完整日期不会同时
进入多个集合，但不保证真实船舶身份隔离。

执行前必须生成不可修改的审计报告并满足以下条件：

1. 下载文件大小、MD5 和 Zenodo 记录一致；
2. `file_location` 去重后再与官方 split 文件取交集，未分配路径和重复行只记录、不擅自加入；
3. 只接纳 `frames == 48000 × 10`、48 kHz、1 或 2 声道的 development 文件；双声道固定读取
   channel 0，绝不做声道平均；
4. `event_time` 可转换为 UTC 日期，且同一 UTC 日期不会跨 development fold；
5. 每个 fold 都含四个目标类别，两个站点和多个日期；
6. 统计每类的日期数、站点数、距离分布、活动状态和缺失字段。

本实验可以验证跨水域、跨日期条件下的 10 s 注意力增益；它不能单独验证未见真实船舶泛化，
也不能等价验证 DeepShip 20 s 长上下文。公开片段通常并不时间相邻，禁止把两个 10 s 文件直接
拼成伪 20 s 样本。

## 2. 四分类映射与样本过滤

统一使用 DeepShip 的四个标签名：

| Belgian 元数据类别 | 统一标签 |
| --- | --- |
| `Cargo` | `Cargo` |
| `Tanker` | `Tank` |
| `Passenger` | `Passenger` |
| `Tug`、`Towing`、`Large-Towing` | `Tug` |

其余船型不参与本实验。主分析只使用目标船距离不超过 5 km 且关键元数据完整的样本，以减少
“最近 AIS 船舶并不是主要声源”的弱标签风险。不得根据音频频谱或模型预测事后删除困难样本。
同时输出 0–1、1–2、2–3、3–5 km 分层指标，检查模型差值是否只是类别距离分布造成。

开发阶段不使用静态的“四类各取同样一批文件”。训练集采用动态平衡，使每轮四类样本数相同，
但让 Cargo/Tank 的不同片段跨轮轮换，从而保留其多样性。validation 和最终 test 均保留全部合格
样本，不下采样，使用 macro-F1 处理类别不均衡。

## 3. 冻结划分

1. 官方 test 从第一次数据处理起封存，不参与划分、选模、阈值、早停或配方修改。
2. 官方 train 与 validation 的合格样本构成 development 候选池；凡 UTC 日期也出现在官方 test
   的候选样本一律从 development 排除，确保最终 test 与开发阶段连日期也不共享。
3. 在 development pool 上生成三个冻结的 date-disjoint folds。分组键为 UTC `calendar_date`；
   同一天两个站点的片段必须在同一 fold，避免共享天气、交通和传播条件跨集合泄漏。
4. 用确定性搜索同时平衡四类样本数、四类日期覆盖和两个站点，不以模型结果选择 split。
5. manifest 记录原始元数据哈希、官方 split 哈希、类别映射、距离门、fold seed、每个文件的日期和
   站点。训练时重新验证哈希和组间零交叉。

三个 fold 的作用是复核日期分配随机性，不把其中任意一个 fold 包装成“最佳划分”。正式结果对
三个 fold 和三个模型 seed（42/43/44）全部汇总。

### 3.1 冻结元数据审计结果

元数据共 28,341 行、28,144 个唯一 `file_location`；197 个重复路径的监督字段一致。官方三个
列表共覆盖 27,554 个唯一路径，其中 58 个路径跨官方 split，已从所有集合整体排除，而不是任意
归入一个集合。5 km 四分类主集共有 13,627 条记录；为保证最终 test 与 development 日期也隔离，
另从 development 排除了与 test 共用四个 UTC 日期的 377 条候选记录。

音频准入前 development 为 11,806 条；精确 10 s 准入后保留 11,533 条（97.69%），其中 5,710 条
单声道、5,823 条双声道。类别为 Cargo 7,545、Tank 3,619、Passenger 173、Tug 196。official test
音频没有被打开或据此筛选，继续保持封存。

重新冻结后的三个 fold 分别使用 22、24、23 个 validation 日期；每个 fold 的 train/validation 均含
四类、两个站点且日期零交叉。Passenger 和 Tug 明显稀少，因此训练使用预注册的动态平衡，验证集
不做下采样。所有 manifest、源文件哈希和零泄漏报告保存在
`protocols/belgian_attention_v1/`，正式训练不得依据结果更换 fold。

## 4. 输入和模型

两个模型共享完全相同的输入：只使用精确 10 s 波形；单声道读取其唯一声道，双声道固定读取
channel 0，再重采样到 16 kHz，然后计算
64-bin log-Mel，`n_fft=1024`、`win_length=1024`、`hop_length=512`，并使用同一归一化。禁止为 G1
延长、拼接或额外增强输入。

| 模型 | 结构 | 研究作用 |
| --- | --- | --- |
| G0 | 现有 MA-CNN-A，约 532k 参数 | 原始专用 CNN 基线 |
| G1 | 相同 CNN，在 `refine_time/refine_freq` 后、全局池化前加入现有 shared temporal axial attention，约 694k 参数 | 检验增加全局时间交互后的变化 |

按照当前决策，Belgian 主实验不运行 G0-C。此前四分支 CNN 扩容几乎没有增益，可作为容量解释的
旁证，但不等同于参数严格匹配对照。因此最终表述只能是“G1 增强模型相对原始 CNN 是否改善”，
不能声称收益已被唯一归因于 self-attention。只有 G1 出现清晰且稳定的正结果、论文评审确实要求
严格容量归因时，才把 G0-C 作为一次补充实验，而不是默认主线。

## 5. 训练采样和优化

每个训练 fold 使用相同的 `class_date_balanced_dynamic` 采样器：

1. 四个类别获得相同配额；每类每轮配额取该 fold 中 Passenger 与 Tug 可用数的较小值；
2. 类内先尽量均匀分配到不同 UTC 日期，再从日期内无放回抽取片段；
3. 稀有类尽量每轮覆盖一次，Cargo/Tank 在不同轮轮换，不永久丢弃大量类内多样性；
4. 每轮保存类别、日期、站点、距离 bin、唯一文件数和重复数审计。

G0 与 G1 固定相同训练配方：AdamW、初始学习率 `3e-4`、weight decay `1e-2`、BF16、有效 batch
32、梯度裁剪 1.0、5 轮 warmup 后 cosine decay、最多 50 轮、patience 8、min delta 0.005。
初始目标为物理 batch 16、梯度累积 2；若统一 smoke 发现显存不合适，只能在正式结果产生前把
两个模型共同改成 batch 8、累积 4。不得只为 G1 调学习率、正则化或早停。

主实验不做水声信道仿真，也不加入模型特有增强。若未来增加轻量 SpecAugment，必须作为 G0/G1
共同的独立配方，不能用它挽救单个模型。

## 6. 选模、指标和统计检验

选模指标为 validation **date-balanced macro-F1**：每个 UTC 日期总权重相同，再由加权混淆矩阵
计算四类 macro-F1。它避免某个录音量特别大的日期主导 checkpoint。同步报告：

- 普通 clip macro-F1、accuracy、balanced accuracy、交叉熵；
- date-balanced macro-F1 和每日期分布；
- Cargo/Tank/Passenger/Tug 的每类 precision、recall、F1；
- 两站点及四个距离 bin 的分层结果；
- 参数、FLOPs、峰值显存、吞吐和训练时间。

所有 G1−G0 比较在相同 fold、model seed 和 validation 文件上配对。最终运行 50,000 次分层 cluster
bootstrap，以 UTC 日期为重采样单元；同时报告三个 fold 的均值方向和不同模型 seed 的方差。

把 Belgian development 结果称为“稳定正增益”至少要求：

1. G1−G0 的 date-balanced macro-F1 总体均值不少于 +1 pp；
2. 95% bootstrap 区间下界大于 0；
3. 三个 fold 的 seed 均值至少两个为正；
4. 提升不只来自单个距离 bin，且任何一个稀有类没有超过 3 pp 的系统性退化。

只有 development 比较完全冻结后，才以各模型预先确定的规则在完整 development pool 重新训练，
并各自对官方 test 评测一次。官方 test 只提供最终外部复核，不回流到配方或模型选择。

## 7. 执行顺序与停止门

1. **M0 元数据准入**：下载前先冻结 URL、大小、MD5；下载后完成去重、类别、日期、站点、距离和
   官方 split 审计。
2. **M1 manifest**：生成三个 date-disjoint development folds 和一个封存 test manifest；运行泄漏
   测试并提交 manifest/hash，不提交音频。
3. **M2 工程 smoke**：G0/G1 各跑一个真实数据 smoke，核验梯度、resume、动态采样审计、指标和
   `test_evaluated=false`。
4. **M3 正式矩阵**：严格按 fold 顺序运行 G0 seed 42/43/44，再运行同 fold 的 G1 seed
   42/43/44；一次只启动一个正式训练，共 18 个单元。
5. **M4 development 分析**：完成日期 cluster bootstrap 和分层误差分析，不读取 test。
6. **M5 一次 test**：只有代码、配方、选择规则和结论模板全部冻结后执行。

若元数据审计发现 Passenger/Tug 在任一 development fold 或封存 test 中少到无法计算稳定四类
指标，先暂停并等待已申请的五天子集，不把距离门逐步放宽到“刚好得到想要的结果”。若 5 km
主集仅样本量偏小但三个 fold 仍有四类，则保留主分析，并把所有距离的结果作为预先标记的敏感性
分析，而不是替换主结果。

## 8. 可能形成的结论

| DeepShip 重复结果 | Belgian G1−G0 | 可支持的结论 |
| --- | --- | --- |
| 稳定为正 | 稳定为正 | 当前轻量全局时间注意力在两个水域、不同隔离单位和 10/20 s 上都表现出可复现价值 |
| 稳定为正 | 无效或为负 | 注意力收益依赖 DeepShip 的时长、标签或环境，不能宣称普遍有效 |
| 无效或不稳定 | 稳定为正 | DeepShip 的船数/划分可能限制结论；注意力在第二个、更大日期覆盖的数据上仍值得研究 |
| 两者均无效或为负 | 无稳定收益 | 有力反对当前 G1 设计，但仍不能外推成“所有 Transformer 都不适合水声分类” |

如果 Belgian 只得到点估计上升但区间跨零，结论是数据证据不足；等待五天连续子集增加独立日期、
连续上下文和可核验身份，而不是继续搜索 G1 层数、头数或学习率。

如果 Belgian 与 DeepShip 重复矩阵都不支持稳定增益，则停止 G 架构搜索，不运行 G2、更多层/头、
只为 G1 定制的优化器或用 official test 挽救结论。项目进入收尾：冻结表现最稳健的轻量 CNN，完成
类别/站点/距离误差分析、参数/FLOPs/吞吐对照和复现实验报告。此时可得出的结论是“当前轻量全局
时间注意力在两套受控水声四分类实验中没有显示稳定优势”，而不是“所有 Transformer 对水声都无效”。
已申请的五天连续子集只在需要研究真实身份隔离或更长连续上下文时作为一个预注册的补充问题，
不再承担为 G1 寻找正结果的职责。

## 9. 训练健康度诊断（`belgian_training_sanity_v1`）

18 个 development 单元完成后，G0/G1 的 date-balanced macro-F1 分别为 0.2561/0.2479，
G1−G0 为 −0.81 pp（95% 区间 −1.78～+0.10 pp）。同时两个模型最终训练准确率仅约 36%～37%，
说明原矩阵首先受到共同欠拟合混杂，不能仅凭该结果判断注意力结构本身。原矩阵及其产物保持冻结，
不覆盖、不续训，也不读取 official test；另开以下一次性诊断协议：

1. **train-only 归一化**：仅使用 fold1 的训练文件计算一个全局 log-Mel mean/std，记录原 manifest
   哈希和统计报告哈希；validation 和 official test 均不得参与统计。
2. **小样本记忆门**：从 fold1 train 按 `SHA256(seed=42, relative_path)` 确定性选择每类 32 条，
   G0 使用 AdamW、学习率 `1e-3`、无 weight decay，最多 1000 个 optimizer steps。连续两次检查均须
   达到 train accuracy ≥98%、macro-F1 ≥98%、CE loss ≤0.10。失败即说明管线或模型学习能力仍有
   问题，暂停，不启动新的正式训练。
3. **全数据 G0 健康度实验**：只有上一步通过才运行 fold1/seed42/G0。每轮遍历所有 train 文件
   恰好一次并确定性 shuffle，不再用最稀类把每轮裁成 456 条；只使用一种不均衡校正——
   `beta=0.999` 的 effective-number class-weighted CE。G0/G1 共用的输入改为 train-fold 全局标量
   标准化。优化器仍为 AdamW、学习率 `3e-4`、有效 batch 32、BF16；warmup 缩短为 1 个完整 epoch，
   早停从 epoch 5 才开始，patience 8、min delta 0.002，最多 30 epoch。每轮新增 train macro-F1
   与四类 train F1 审计。
4. **G0 学习健康门**：最终 train accuracy ≥70%、train macro-F1 ≥60%，且 Passenger/Tug 的
   train F1 各 ≥40%。未通过则判定公开 Belgian 四分类弱标签/域偏移在当前规模下不适合作为可靠
   注意力验证，停止 Belgian 模型搜索并整理负结果。
5. **匹配 G1 仅在门后发生**：G0 通过后，先冻结其学习曲线和验证结果，再用完全相同的 full-data、
   loss、normalization、优化预算运行一个 fold1/seed42/G1。若单对结果没有至少 +1 pp 的
   date-balanced macro-F1，停止；若有，再预注册是否扩展到多 seed/fold，不能直接读取 test 或
   事后为 G1 单独调参。

该诊断回答“此前阴性是否主要由明显欠训练造成”，不是新的架构搜索，也不能把单 fold/single seed
写成最终外部泛化结论。唯一输出根使用 `belgian_training_sanity_v1`，不得复用
`belgian_attention_v1`。

## 10. 数据来源

- Zenodo 数据集：<https://zenodo.org/records/17233667>
- 作者数据说明：<https://github.com/woutdecrop/audio_vessel_distance_categorizer/blob/main/README_dataset.md>
- 关联论文 DOI：<https://doi.org/10.1109/JSTARS.2025.3593779>
