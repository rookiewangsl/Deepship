# Oceanship-FG 回溯 ONC 连续录音计划

最后更新：2026-08-29  
状态：O1/O2 探针完成；确认 FG 船舶标签与 ONC 水听器存在结构性时空错配，停止监督数据扩充和
批量音频下载。完整证据见
[Oceanship-FG/ONC 时空一致性验证](../research/oceanship_onc_spatiotemporal_validation.md)。

## 1. 角色与研究问题

这条路线的首要用途是扩充有 MMSI 身份标注的监督训练/开发数据，并在独立船舶数量增加后复核
20 s 全局注意力是否有效。独立外部评测必须使用另一个满足身份、连续性和标签纯度要求的数据源，
避免同一个外部数据集同时承担训练扩充和最终验证。

Oceanship-FG 公开的是约 4～5 s 样本，不能直接检验 L20。回溯目标是根据 FG 文件名中的事件
UTC 时间和 MMSI，定位 ONC 原始水听器归档，重建真实连续、标签纯度经过复核的 20 s 波形。

## 2. 已确认的元数据事实

- 服务器已有 FG train/test CSV 和原作者抓取代码，不需要先下载整个 Oceanship 发布包；
- 原始抓取代码使用水听器 `ICLISTENAF2523`、`ICLISTENAF2556`；
- 四个目标类共有 27,193 行、19,206 个事件、414 个无类别冲突的 MMSI：
  - Cargo：9,662 行、6,608 个事件、221 个 MMSI；
  - Passenger：5,527 行、4,268 个事件、60 个 MMSI；
  - Tanker：152 行、114 个事件、19 个 MMSI；
  - Tug：11,852 行、8,216 个事件、114 个 MMSI；
- 官方 train/test 共有 170 个目标类 MMSI 跨集合，不能用于严格未见船舶评测，必须重新按 MMSI
  冻结划分；
- `ais_timestamp` 与 `wav_path` 文件名开头时间的中位差约 42,159 s，因此它不是片段定位时间；
- ONC 查找锚点必须使用 `wav_path` 中的 `YYYYMMDDTHHMMSS.sssZ` 事件时间，并对两个候选水听器
  都进行短区间查询；
- 原始代码不同阶段存在 1/11 km、以及可配置 inclusion/exclusion 半径并存的情况。重建数据不能
  直接继承“单船纯净”结论，必须使用对应时段 AIS 重新审计。

## 3. 分阶段准入门

### O0：离线元数据审计（已完成）

使用 `scripts/prepare/audit_oceanship_fg_onc.py` 解析文件名时间、事件、MMSI 和类别，生成：

- `metadata_audit.json`；
- 每类 3 个不同 MMSI、尽量不同日期的 `onc_probe_candidates.csv`。

这一步不访问 ONC，也不下载音频。

### O1：ONC 归档只查询探针

在独立 Python 环境安装 `requirements-onc.txt`，通过环境变量传入个人 `ONC_TOKEN`。运行
`scripts/prepare/probe_oceanship_onc_archive.py`，对 12 个候选事件、两个水听器各查询事件前后
5 min 的 WAV 文件名。默认只写归档索引，不下载音频。

准入条件：

1. 四类均至少有一个事件能唯一定位到一个水听器；
2. 归档文件时间覆盖候选 20 s；
3. 文件名时间与 FG 事件时间的关系在多个日期上一致；
4. 不依赖公开仓库中硬编码的第三方 token。

若大部分事件无法定位，先检查设备部署、时区和 FG 文件名语义，不扩大查询范围或整库下载。

实际结果：12/12 候选均能在 `ICLISTENAF2523` 定位到覆盖事件时间的 5 min WAV；这只证明时间
索引可恢复，不证明 MMSI 标签与录音地点一致。

### O2：极小音频重建原型

仅下载 O1 命中的少量归档 WAV，目标为每类至少 2 个 MMSI、每船 1～2 个连续 20 s 窗口。
重建时保留原始文件名、设备、UTC 起止、采样率、声道、截取偏移和 SHA-256。

验收条件：

1. 解码后连续覆盖不少于 20 s，没有拼接缺口、重复区间或补零；
2. 截取时间与 FG 事件时间一致；
3. 用 AIS 重新验证目标 MMSI、目标类别和近场其他船舶；
4. 人工试听/频谱抽查没有明显文件边界、静音或损坏；
5. 原始采样率归档，训练副本再统一为单声道 16 kHz；
6. 原始归档与派生片段分目录存放，任何派生文件都能从 manifest 重建。

实际结果：四类各下载一个 300 s、32 kHz、单声道 PCM-24 WAV，目标 20 s 均可连续解码且非静音；
但 AIS 复核未通过。水听器在 Campbell River，而 FG 使用的 AIS 源在 Vancouver，两者约相距
171 km。12 个分层候选的目标 MMSI 在近场 AIS 中出现 0/12，在 Vancouver AIS 中出现 11/12；
已匹配目标船距水听器 163.6～190.7 km。故不能把可解码音频视为对应目标船的监督样本。

### O3：规模与类别可行性门

先枚举可重建事件，不下载全部音频。按 MMSI、类别、设备、日期和可用连续时长报告容量。Tanker
只有 19 个 MMSI，是最可能限制严格四分类划分的类别。

只有以下条件满足才进入批量下载：

- train/validation/test 能按 MMSI 严格隔离且每个集合均有四类；
- 每类 validation/test 有足够独立 MMSI，不通过窗口过采样伪造样本量；
- 单船纯度审计通过；
- 可重建 20 s 的比例足以支持 G0/G0-C/G1 同协议比较；
- 先估算下载文件数、原始体积和派生体积，并设置并发/速率上限。

当前结论：未通过“目标 MMSI 与水听器近场位置一致”和“单船纯度”两项前置条件，O3 不执行。
不得通过扩大样本量来补救结构性标签错配。

### O4：受控模型复核

DeepShip-only 3×3×3 重复完成并冻结模型后，再在扩充数据上运行同一 G0/G0-C/G1 结构。三者
必须使用同一 MMSI split、同一 20 s manifest、相同优化预算与选模指标。外部评测数据不参与训练
或超参数选择。

## 4. 数据与凭证安全

- ONC 官方客户端要求个人 Oceans 3.0 token；只从 `ONC_TOKEN` 环境变量读取；
- 不使用或传播 Oceanship 公开仓库中硬编码的 token；
- token 不写入命令行日志、配置、manifest 或 Git；
- 所有 ONC 客户端异常在写入终端或日志前必须移除 URL 查询参数中的 token；若第三方客户端异常
  意外打印 token，应立即吊销并重新生成，不继续使用已暴露凭证；
- 初始探针单线程/低并发，遵循 ONC 关于避免大量并发下载的建议；
- 当前 DeepShip GPU 重复实验运行期间，只做元数据和小样本网络工作，不进行大规模下载、解码或
  `/home` 高强度 I/O。

## 5. 决策解释

- O1/O2 通过：说明 Oceanship-FG 可以成为 ONC 连续 20 s 的标签索引，随后再评估实际规模；
- 能定位但 AIS 纯度不足：只可作为弱标签或无标签水声数据，不进入严格监督四分类；
- 只能恢复短片段：可用于独立船数扩充，但不能用于证明长上下文注意力；
- 无法稳定定位：停止回溯，不把整库下载作为补救，转向其他有身份长录音。

本次命中第二种情况，但错配程度比一般弱标签更强：FG 类别标签不应与恢复出的 Campbell River
音频配对。已下载约 110 MB 音频、504 MB Vancouver AIS 和 427 MB 近场 AIS 仅作为可复核探针
保留，不再扩容。若未来使用这些水声，只能作为无标签领域音频；若按近场 AIS 重新构造标签，则
属于一个新的数据集建设任务，必须重新进行船型映射、近场排他过滤与人工审计，不能沿用 FG 标签。

## 6. 官方依据

- [ONC Python 客户端](https://github.com/OceanNetworksCanada/api-python-client/)
- [ONC archived-files 示例](https://oceannetworkscanada.github.io/api-python-client/Code_Examples/Download_Archived_Files.html)
- [Oceans 3.0 API](https://oceannetworkscanada.github.io/Oceans3.0-API/)
