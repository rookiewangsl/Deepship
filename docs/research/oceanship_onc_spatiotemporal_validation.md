# Oceanship-FG/ONC 时空一致性验证

最后更新：2026-08-29  
结论：Oceanship-FG 文件名可以定位 ONC 原始音频，但公开 MMSI/类别标签与该水听器不在同一
海域，不能用于构造严格监督的连续 20 s 船舶分类数据。

## 1. 验证问题

Oceanship-FG 公开样本约 4～5 s。原计划根据文件名 UTC 和 MMSI 回溯 ONC 连续归档，扩充独立
船舶数量，并复核长上下文全局注意力。验证必须同时满足两件事：

1. 事件时间能定位并连续截取不少于 20 s 的原始水声；
2. 标签 MMSI 在事件时刻确实位于水听器附近，且近场竞争船数量可接受。

仅满足第一项不能形成有效监督样本。

## 2. 数据源和抽样

- 从四个目标类各取 3 个不同 MMSI，共 12 个候选，覆盖 11 个日期；
- 音频设备：`ICLISTENAF2523`；
- FG 原代码使用的 AIS 设备：`DIGITALYACHTAISNET1302-0097-01`；
- 与水听器同址的 AIS 设备：`SHINEMICRORADARPLUSAIS151`；
- 对每个候选解码事件前后 5 min 的 AIS 位置报告；
- 距离使用球面 Haversine 距离；目标是否出现以 MMSI 精确匹配判定。

服务端可复核产物：

- `analysis/oceanship_onc_reconstruction_v1/onc_probe_candidates.csv`；
- `analysis/oceanship_onc_reconstruction_v1/nearfield_ais_window_audit_v1.json`；
- `analysis/oceanship_onc_reconstruction_v1/distant_ais_window_audit_v1.json`；
- `analysis/oceanship_onc_reconstruction_v1/onc_archive_index_probe_v1.csv`；
- `analysis/oceanship_onc_reconstruction_v1/onc_nearfield_ais_archive_probe_v1.csv`。

## 3. 官方部署位置

ONC deployment 元数据显示，研究日期内：

| 设备 | 位置 | 纬度 | 经度 | 与水听器关系 |
|---|---|---:|---:|---|
| `ICLISTENAF2523` | CRIP, Campbell River | 50.020767 | -125.235350 | 水听器 |
| `SHINEMICRORADARPLUSAIS151` | CRSS | 50.020867 | -125.235367 | 约同址 |
| `DIGITALYACHTAISNET1302-0097-01` | IONA, Vancouver | 49.216063 | -123.205482 | 相距约 171.4 km |

近场 AIS 日文件中的 GPS 句也报告约 `50.02086, -125.23538`，与 deployment 信息一致。

## 4. 音频定位结果

- 12/12 候选均在 `ICLISTENAF2523` 找到覆盖事件时间的归档 WAV；
- 每个候选对应一个唯一 5 min 文件，且事件后至少剩余 20 s；
- 下载四类各一个原始文件，均为 300 s、32 kHz、单声道 PCM-24；
- 四个目标 20 s 窗口均为有限值、非静音，没有解码或连续性问题。

因此，FG 文件名中的 UTC 可以作为 Campbell River 归档的时间索引。

## 5. AIS 对照结果

| 检查 | 近场 AIS（Campbell River） | FG 使用的 AIS（Vancouver） |
|---|---:|---:|
| 可用候选 | 12/12 | 12/12 |
| 目标 MMSI 出现 | 0/12 | 11/12 |
| 每窗口不同位置 MMSI | 42～57 | 178～252（四类首轮样本） |
| 近场 11 km 内不同 MMSI | 32～42 | 不适用于水听器近场判断 |

Vancouver AIS 中 11 个目标的最近位置报告与事件时间绝对差中位数为 4.43 s；其中 8/11 不超过
10 s。它们相对 Campbell River 水听器的距离为 163.6～190.7 km，中位数 174.1 km。唯一未在
±5 min 找到位置报告的是一个 Tanker 候选，不改变整体结论。

近场 12 个窗口共成功解码 11,963 条位置报告。部分多片静态 AIS 消息被单句审计器计为解码错误，
但位置消息为单片消息，且每个窗口均有大量有效船位，因此不会造成目标 MMSI 0/12 的假阴性解释。

## 6. 结论与项目影响

最符合证据的解释是：FG 事件时间和 MMSI 来自 Vancouver AIS 流，而同一 UTC 被用于截取约
171 km 外 Campbell River 水听器的音频。目标船不可能被视为这些音频的可靠声源，且录音附近
同时存在大量其他船舶。

因此：

- 终止 Oceanship-FG → ONC 20 s 的监督数据扩充路线；
- 不进入全量归档枚举或批量音频下载；
- 不用这些 FG 标签训练或评测 G0/G0-C/G1、Conformer 或 CNN；
- 已下载探针保留用于复核，不纳入模型数据 manifest；
- Campbell River 原始水声仍可作为无标签领域音频；
- 依据近场 AIS 从零构建新标签在技术上可能，但因 10 min 窗口内 11 km 范围已有 32～42 个
  不同 MMSI，必须另做严格近场排他规则、船型注册表关联和人工审计，不能视为低成本补救。

这项结果不回答“全局注意力是否有效”。它只否定了 Oceanship-FG 回溯作为该问题外部监督验证集
的有效性。DeepShip-only 重复实验继续按原计划完成；外部有船标注验证应转向其他能证明水听器、
AIS 与船型标签同址同刻的数据源。

## 7. 可复现命令入口

- 元数据抽样：`scripts/prepare/audit_oceanship_fg_onc.py`；
- 归档索引：`scripts/prepare/probe_oceanship_onc_archive.py`；
- AIS 窗口审计：`scripts/prepare/audit_oceanship_ais_windows.py`；
- 可选依赖：`requirements-onc.txt`。

验证脚本只解码 validation probe，不读取 DeepShip test，也没有改变后台训练任务。
