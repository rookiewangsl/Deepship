# DeepShip 录音级与船只级隔离训练路线

## 目的与边界

历史 `95.15%` 是 V1 片段级随机划分结果，不作为新录音或新船只泛化结论。本路线新增两种
隔离评测，并把“数据协议生成”和“GPU 训练”解耦：本地负责审核并冻结数据协议，远程
Windows/Linux 主机只读取协议训练。本文保留原始路线，同时记录已经完成的实施结果。

## 当前实施状态（2026-08-12）

- 已冻结 `configs/experiments/isolation_comparison_v1.json`；三种协议均为每类
  `3500/1000/500`，即 70%/20%/10%，正式 model seed 为 `42/43/44`。
- 已审计 T7 的 609 条 WAV，并完成全部音频内容 SHA-256；发现 6 对完全重复录音，录音级协议
  按内容哈希共同分组，避免副本跨集合。
- 船名映射中 603 条录音可纳入，6 条未解析 Tank 录音排除；得到 247 个船名组。当前第三组
  仍严格命名为 `vessel_name_disjoint`，不声称 MMSI/IMO 物理船只隔离。
- 已生成并验证三个各含 20,000 个片段的冻结 manifest；所有路径均为 POSIX 风格相对路径，
  不包含 macOS 挂载点或 Windows 盘符。
- 已实现 manifest 驱动训练、checkpoint/resume、分层评估、Windows PowerShell 操作脚本和
  三 seed 完整矩阵的一致性检查/汇总脚本。
- Windows 三组 smoke test 已通过。随后在 RTX 4070 上完成三种协议的 model seed 42/43，
  共六个正式 run，均有完整 checkpoint、预测、指标、环境和 `run_complete.json`。
- 当前 segment Accuracy 为：`segment_level` 97.30±0.21%、`recording_disjoint`
  66.20±0.49%、`vessel_name_disjoint` 50.98±0.18%。主要聚合 Accuracy 为 recording
  70.19±4.08%、vessel-name group 53.13±4.42%。
- 由于时间原因未运行 seed 44。当前报告明确按两个 seed 统计；若需要满足原冻结计划和严格
  自动汇总器的完成标准，再补 seed 44。

## 三种协议

| 协议 | 分组键 | 使用数据 | 可以声称的结论 |
|---|---|---|---|
| `segment_level` | 片段 | 审计 609 条，manifest 选中 596 条录音 | 片段级基线复现；不代表独立录音泛化 |
| `recording_disjoint` | 音频内容组（相对路径＋重复哈希） | 审计 609 条，manifest 选中 594 条录音/588 组 | 对未见录音内容组的泛化 |
| `vessel_name_disjoint` | 经审核的 `vessel_key` | 603 条可纳入，manifest 选中 581 条录音/244 名称组 | 对未见规范船名组的泛化 |
| `physical_vessel_disjoint` | MMSI / IMO / 发布方确认的物理船只 ID | 身份核验完成后确定 | 对未见物理船只的严格泛化 |

`vessel_name_disjoint` 不是 `physical_vessel_disjoint` 的同义词。没有可核验的 MMSI、IMO 或
发布方身份映射时，不能将船名组隔离结果表述为绝对物理船只隔离。

## 目标代码结构

```text
原始数据 + 船只元数据
    |
    +-- 本地：协议编译与审核（一次）
    |     +-- recording_vessel_manifest.csv
    |     +-- split_manifest.json
    |     +-- validation_report.json
    |     +-- exclusions.csv
    |
    +-- 远程：训练（每个 seed 一次）
          +-- 读取固定 split_manifest.json
          +-- 通过本机 --data-root 定位 WAV
          +-- 输出 checkpoint、日志、指标、图和环境信息
```

训练脚本不得在运行时重新随机划分数据；`seed` 仅影响模型初始化、DataLoader 顺序和训练随机性。

## 原始实现计划与剩余工作

以下数据协议、训练和评估工作已完成；保留细节用于复核。仍未完成的是 MMSI/IMO 身份核验、
physical-vessel manifest、seed 44，以及部署性能测试。

### 1. 身份清单与准入规则

1. 保留并扩展 `recording_vessel_manifest.csv`：使用相对路径、类别、规范船名、`vessel_key`、
   匹配方法、置信度、身份审核状态和证据来源。
2. 完成 6 条未解析录音的人工或外部数据核验；不能唯一确认身份的记录从船只级协议排除，
   而不进行猜测性归属。
3. 建立可审查的别名/MMSI/IMO 映射。若获得稳定物理 ID，新增 `physical_vessel_id` 并使
   同一套划分器以该字段分组。
4. 报告同一组跨类别、重复 ID、冲突标签和低置信度匹配；严格协议遇到未处理冲突应停止，
   不能静默继续。

### 2. 冻结的 group-aware 划分器

新增独立的准备脚本（拟定为 `scripts/prepare/build_deepship_split.py`）：

1. `recording_disjoint`：默认每条 WAV 是一个 group；内容 SHA-256 完全相同的不同路径合并为
   同一 group，避免副本跨集合。审计覆盖全部 609 条录音。
2. `vessel_name_disjoint`：同一 `vessel_key` 下所有录音强制进入同一个 partition。
3. 按类别做 group-aware 的 train/validation/test 候选容量分配，固定目标为 70%/20%/10%。
   组不可拆分；完成 group 分配后，在各集合内部确定性抽取每类 `3500/1000/500` 个片段。
4. 在组分配后才从录音产生 3 秒片段，确保同一 WAV 的任何片段绝不跨 partition。
5. 输出不可变 `split_manifest.json`、统计报告、排除清单和输入/输出 SHA-256。

所有可移植 manifest 只记录 `relative_path`，绝不记录 `/Volumes/...`、Windows 盘符或其他
机器相关的绝对路径。

### 3. 强制验证

每次生成协议必须验证：

1. train、validation、test 的 group 集合两两没有交集；
2. 同一 `relative_path` 只能出现一次且只属于一个集合；
3. 每个入选 recording 恰好有一个类别和一个有效 group；
4. 每类在各集合均有样本；
5. 对船只级协议，所有未解析、低置信度或冲突记录都有明确的纳入/排除理由；
6. 必须使用完整音频内容 SHA-256 检查不同路径下的完全重复音频；录音级协议按相同内容哈希
   共同分组，防止副本跨集合。

验证失败时不生成可供训练使用的协议。

### 4. 训练入口改造

`TrainConfig` 和 `scripts/train/train_deepship_macnna.py` 将增加：

- `--split-manifest`：必填于隔离协议；训练仅读取该文件；
- `--data-root`：当前机器的数据集根目录；
- `--num-workers`：按远程机器硬件设置；
- `--protocol-name`：用于输出目录与报告标识。

保留现有 paper/segment split 作为复现实验路径，但隔离训练不能调用现有
`build_paper_split()`。

每次训练在输出目录保存：输入 manifest 副本及哈希、完整 CLI/配置、git commit、Python/
PyTorch/CUDA 环境、最佳模型、训练历史、片段级指标、录音级聚合指标和混淆矩阵。

## 已执行的实验顺序

1. 实现并冻结 `recording_disjoint`，先验证全 609 条录音的流程。
2. 审核船名组映射后，实现并冻结 `vessel_name_disjoint`。
3. 各协议先执行一个短 smoke run，确认路径、manifest、GPU 和输出链路无误。
4. smoke run 通过后固定代码 commit 与 manifest；当前实际完成每种协议 2 个训练 seed
   （42/43），原计划的 seed 44 保留为可选补充。
5. 使用验证集早停和选模型；测试集只用于最终评估。比较不同协议时保持模型、特征、epoch
   上限和训练规则一致。
6. 远程完成后，只回传 `runs/<实验名>/`，不回传原始数据集或可重建缓存；本地统一汇总结果。

## 远程 Windows / Linux 训练原则

1. 上传或拉取固定 git commit 与冻结的协议目录；数据集可放在任意本机路径。
2. 训练命令只因 `--data-root`、`--output-root` 和 `--device cuda` 的平台路径不同而不同。
3. 不在远程机器重新生成船只映射或 split manifest；若协议需要变更，应在本地重新审核、
   生成并赋予新的协议版本/哈希。
4. 远程先运行 CPU 或少量 epoch 的 smoke run，再开始正式多 seed 训练。

Windows 示例：

当前训练机的代码位于 `C:\Users\shilongwang\Desktop\Deepship`，T7 数据集根目录为
`D:\ProjectData\Deepship\datasets\DeepShip`。协议文件不记录盘符；若以后盘符变化，只修改
`-DataRoot`。

```powershell
.\scripts\windows\run_formal.ps1 `
  -DataRoot "D:\ProjectData\Deepship\datasets\DeepShip" `
  -OutputRoot ".\runs\isolation_comparison_v1_epoch_output" `
  -Seeds 42,43 `
  -NumWorkers 0
```

完整步骤见 [`windows_training_guide.md`](../guides/windows_training_guide.md)。

## 完成标准

1. 协议文件可在 macOS、Windows 和 Linux 上不修改内容地使用；
2. 任意训练输出能追溯到唯一 git commit、唯一 manifest 哈希和唯一训练配置；
3. 自动化测试覆盖同船多录音、多片段、别名合并、未解析记录和交集检测；
4. 对外报告清楚标示评测协议，不把 recording-disjoint 或 vessel-name-disjoint 混称为物理
   船只完全隔离；当前两个 seed 可报告为阶段性正式结果，补齐 seed 44 后才满足原三 seed
   完成标准。

## 逐项实施计划

以下阶段按顺序执行。每一阶段通过验收后再开始下一阶段，避免在数据协议、训练代码和远程
环境同时变化时难以定位问题。

### 阶段 0：冻结实验定义

目标：在修改代码前固定三组实验中不能变化的条件。

拟定默认值：

- 网络：当前新的 MA-CNN-A 三分支结构；
- 输入：16 kHz、3 秒、不重叠、64 个 Mel bins；
- 三组协议的目标片段预算：每类 train/validation/test 为 `3500/1000/500`，对应
  70%/20%/10%；
- split seed：固定为 `42`；
- 正式训练 model seed：`42/43/44`；
- 优化器、学习率调度、batch size、epoch 上限和 early stopping 在三种协议间一致；
- 第三组在没有 MMSI/IMO 映射前命名为 `vessel_name_disjoint`。

注意：录音级和船名组级不能为了满足片段预算拆分 group。若数据审计证明某类无法达到目标，
应统一降低三种协议该类/partition 的预算，或改为报告全部可用片段；不能只对某一种协议临时
改变规则。

产物：

- 一份版本化实验配置，例如 `configs/experiments/isolation_comparison_v1.json`；
- 网络参数量和关键结构的断言；
- 协议名称、数据预算、split seed、model seed 的书面定义。

验收：任何人只读配置文件即可准确描述三个实验的相同项和唯一差异项。

### 阶段 1：数据与身份审计

目标：确认 609 条 WAV、船名映射和严格协议可使用的实际范围。

工作项：

1. 扫描数据集，生成相对路径、类别、采样率、帧数、时长和可产生的 3 秒片段数。
2. 将扫描结果与 `recording_vessel_manifest.csv` 做一对一连接。
3. 检查重复相对路径、缺失文件、无法读取的 WAV、AppleDouble 文件、类别不一致和同组跨类。
4. 汇总 high/medium/none 置信度、截断船名、别名合并和 6 条未解析录音。
5. 计算每类、每录音、每 vessel group 的可用片段数，验证三组目标预算是否可行。
6. 对全部音频计算内容 SHA-256，检查不同路径下的完全重复文件；已发现的 6 对 Passenger
   重复录音在录音级协议中按内容哈希共同分组，避免副本跨集合。

产物：

- `dataset_inventory.csv`；
- `identity_audit.json`；
- `identity_exclusions.csv`；
- 数据清单哈希。

验收：所有 609 条 WAV 都有唯一审计状态；船名组协议的每条纳入/排除记录都有明确理由；
目标片段预算得到可行性结论。

### 阶段 2：跨平台相对路径数据层

目标：同一份协议不包含机器相关路径，可在 macOS、Windows 和 Linux 使用。

工作项：

1. 数据记录统一保存 POSIX 风格的 `relative_path`，例如
   `Cargo/20171104-1/203623.wav`。
2. `Dataset` 在运行时使用 `data_root / relative_path` 定位音频。
3. 删除新 manifest 中的绝对路径依赖；旧结果读取逻辑如需保留，明确标为兼容模式。
4. 加入路径逃逸检查：拒绝绝对路径和包含 `..`、越出 `data_root` 的记录。
5. 为 Windows 路径、空格路径和不同盘符增加测试。

产物：可移植数据记录结构及相关单元测试。

验收：同一 manifest 在三个操作系统上只改变 `--data-root` 即可定位相同相对文件集合。

### 阶段 3：统一协议编译器

目标：一次性生成三种冻结的、可验证的 split manifest。

拟新增：`scripts/prepare/build_deepship_split.py` 和对应的 `src/data` 划分模块。

工作项：

1. 定义版本化 manifest schema，至少包含：协议名、schema 版本、split seed、源清单哈希、
   相对路径、类别、group key、partition、片段起点和帧数。
2. `segment_level`：从各类候选片段中按固定 seed 选择 `3500/1000/500`。
3. `recording_disjoint`：先把 WAV 分配给 partition，再在各 partition 内抽取目标片段。
4. `vessel_name_disjoint`：先把 vessel group 分配给 partition，再从组内录音生成和抽取片段。
5. group 分配优化目标以“可产生的片段数和类别预算”为主，而不是只按 group 数量平均。
6. 分配算法必须确定性：相同输入和 seed 产生字节级一致的核心 manifest。
7. 同时输出 recording assignment、最终 segment manifest、排除清单和统计报告。

强制验证：

- group、recording、segment 在不该相交的集合间均无交集；
- 所有 segment 都落在对应 recording/group 的 partition；
- 每类实际样本数达到冻结预算；
- train、validation、test 均有四类；
- manifest 哈希可重复计算并一致。

验收：对同一输入连续运行两次得到相同哈希；人为构造泄漏时测试必须失败。

### 阶段 4：训练管线读取冻结协议

目标：GPU 主机只训练，不重新决定数据划分。

工作项：

1. `TrainConfig` 和 CLI 增加 `--split-manifest`、`--protocol-name`、`--num-workers`。
2. 隔离实验必须提供 manifest；训练开始前再次执行完整交集与文件存在性检查。
3. DataLoader 直接使用 manifest 中冻结的 segment 列表。
4. 保存实际命令、完整配置、manifest 副本与哈希、git commit、Python/PyTorch/CUDA 信息。
5. 输出目录存在完整 run 时拒绝覆盖；中断训练使用明确的 `--resume` 恢复。
6. 除最佳 checkpoint 外保存 last checkpoint，包含 optimizer、scheduler、epoch、最佳验证指标
   和随机状态，保证远程中断后可继续。
7. Windows 默认从 `num_workers=0` 开始，正式运行前再通过 smoke test 决定是否提高。

验收：训练 seed 的变化不改变 manifest 哈希；中断并恢复后的 epoch、学习率与早停状态连续。

### 阶段 5：多层级评估

目标：不仅报告片段准确率，还真实反映新录音和新船只表现。

工作项：

1. 保留 segment-level Accuracy、macro-F1、weighted-F1、per-class 指标和混淆矩阵。
2. 将同一 WAV 的片段预测概率平均，生成 recording-level 预测与同类指标。
3. 对 vessel-name 协议增加 vessel-group 聚合指标；聚合时先得到每条录音概率，再对录音等权
   平均，避免长录音因片段多而占过高权重。
4. 每条预测记录相对路径、group key、真实标签、预测标签和概率，便于回溯错误。
5. 测试集只在最佳验证模型确定后评估一次。

产物：`segment_metrics.json`、`recording_metrics.json`、`vessel_metrics.json`、预测 CSV
和对应图表。

验收：聚合指标能由保存的预测文件重新计算得到，且测试 DataLoader 不打乱记录对应关系。

### 阶段 6：自动化测试与本地 smoke test

目标：不依赖本地 GPU，也能验证协议和训练链路。

测试范围：

1. 同一 WAV 的多个片段不能跨集合；
2. 同一 vessel 的多个 WAV 不能跨集合；
3. 别名映射后仍归入同一 group；
4. 未解析、低置信度和冲突记录按配置排除或报错；
5. manifest 确定性、哈希、schema 和路径安全；
6. 使用临时小 WAV 完成一次 Dataset/DataLoader/模型前向；
7. 使用极小数据完成一个 epoch，并验证 checkpoint、resume 和指标输出。

验收：全部单元测试通过；CPU smoke run 完成；无须访问正式测试集即可验证链路。

### 阶段 7：Windows 运行包与操作文档

目标：Windows 主机可按固定步骤安装、检查、训练和恢复。

工作项：

1. 提供基础依赖清单；PyTorch/CUDA 按 Windows GPU 驱动环境单独安装并记录版本。
2. 提供 PowerShell 数据检查、smoke test、单次训练和三 seed 批量训练示例。
3. 训练前检查 `nvidia-smi`、`torch.cuda.is_available()`、数据文件数、manifest 哈希和可用磁盘。
4. 代码与输出放本机 SSD，移动硬盘只读数据；关闭系统睡眠并避免训练时拔盘。
5. 日志实时写文件；每个 run 使用独立目录，失败时保留日志和 last checkpoint。

验收：Windows 先完成每种协议各一个 1 epoch smoke run，并能从 last checkpoint 恢复。

### 阶段 8：正式远程训练（seed 42/43 已完成）

执行顺序：

1. 固定并记录 git commit；push 代码、配置和三个协议，不上传 WAV 或缓存。
2. Windows `git pull` 后再次核对 commit 和 manifest 哈希。
3. 先跑三个协议的 model seed 42；检查 loss、样本数、输出完整性和显存。
4. seed 42 均正常后运行 seed 43；当前因时间限制没有继续 seed 44。
5. 每个 run 完成后执行结果完整性检查；不完整 run 不进入最终统计。
6. 将 run 目录复制回本地，复制后对关键 JSON、预测文件和 checkpoint 校验哈希。

当前验收：6 个正式 run（3 个协议乘 2 个 model seed）均具有完整配置、日志、checkpoint、
预测和指标；三种协议之间除 split manifest 和 model seed 外的训练配置一致。原计划的完整
验收仍是 9 个 run，补齐 seed 44 后达成。

### 阶段 9：结果汇总与结论边界（两 seed 汇总已完成）

目标：生成可复核的三协议对比结果。

工作项：

1. 使用 run 内配置、环境和 manifest 哈希核对当前六个 run 的模型结构、特征、优化配置、
   commit 和协议一致性；完整三 seed 矩阵仍可使用 `scripts/eval/summarize_isolation_runs.py`。
2. 当前分协议统计两个 seed 的均值、样本标准差和单次结果；补 seed 44 后重新生成三 seed
   终稿。
3. 对比 segment、recording、vessel-group 三个层级的 Accuracy 与 macro-F1。
4. 生成统一表格、混淆矩阵和训练曲线；保留失败类别与典型误判的可追溯记录。
5. 明确表述：无隔离结果、录音级隔离结果、船名组隔离结果；只有获得物理 ID 后才能增加
   物理船只严格隔离结论。

当前验收：本文档中的每个数字都能追溯到某个 run、manifest 哈希和预测文件。
文献比较见 [`deepship_literature_benchmarks.md`](../research/deepship_literature_benchmarks.md)。

结果复制回本地后的命令：

```bash
python scripts/eval/summarize_isolation_runs.py \
  --runs-root /path/to/isolation_comparison_v1
```

输出位于 `<runs-root>/summary/`，包含逐 run CSV、分协议均值/样本标准差 CSV、JSON、聚合混淆
矩阵和 Markdown 对比表。

## 原始逐次实施顺序

实际协作时按以下九个任务逐一进行，每次只推进一项：

1. 冻结实验配置并完成数据/身份审计；
2. 改造相对路径数据层；
3. 实现和测试统一协议编译器；
4. 生成并审核三份冻结 manifest；
5. 改造训练管线及 checkpoint/resume；
6. 实现 recording/vessel 多层级评估；
7. 完成本地测试和 CPU smoke run；
8. 准备 Windows 指南并执行远程训练；
9. 回传结果并生成最终对比报告（两 seed 版本已完成；三 seed 版本待可选 seed 44）。
