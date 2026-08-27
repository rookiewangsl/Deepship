# DeepShip 三协议 Windows GPU 训练指南

最后更新：2026-08-12

本指南用于在 Windows 电脑上读取挂载的 T7 数据，运行新 MA-CNN-A 网络的片段级、录音级和
船名组级三组对比实验。代码、协议和训练输出放 Windows 本机 SSD；T7 只读原始音频。

当前状态：三组 smoke 已通过，seed 42/43 的六个正式 run 已完成并回传本地。seed 44 因时间
原因未运行。

## 1. 本次实际目录

```text
C:\Users\shilongwang\Desktop\Deepship                         # Git 仓库（本机 SSD）
C:\Users\shilongwang\Desktop\Deepship\runs                    # 训练输出（本机 SSD）
D:\ProjectData\Deepship\datasets\DeepShip                      # T7 原始数据
```

本次 T7 是 `D:`。以后盘符仍可能变化，每次接入后先确认：

```powershell
Get-Volume | Select-Object DriveLetter, FileSystemLabel, HealthStatus, SizeRemaining
```

检查以下四个目录确实存在：

```text
D:\ProjectData\Deepship\datasets\DeepShip\Cargo
D:\ProjectData\Deepship\datasets\DeepShip\Passenger
D:\ProjectData\Deepship\datasets\DeepShip\Tank
D:\ProjectData\Deepship\datasets\DeepShip\Tug
```

三个 manifest 只保存 `Cargo/...` 等相对路径，因此改变盘符不需要重新生成协议，只需要修改
命令中的 `-DataRoot`。

## 2. 拉取代码

首次 clone：

```powershell
Set-Location C:\Users\shilongwang\Desktop
git clone git@github.com:rookiewangsl/Deepship.git
Set-Location C:\Users\shilongwang\Desktop\Deepship
```

已有仓库更新：

```powershell
Set-Location C:\Users\shilongwang\Desktop\Deepship
git pull origin main
git rev-parse HEAD
git status --short
```

正式训练前记录 commit，并保持工作区干净。本次六个 run 使用 commit
`bf1ed7d0d5a911b2929ba02ec7396efb4adc774a`。

## 3. Python 环境

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch/CUDA 安装应以 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 为准，
并确保 `torch` 与 `torchaudio` 版本匹配。本次成功环境是 Python 3.14.6、PyTorch
2.11.0+cu130、CUDA 可用、NVIDIA GeForce RTX 4070；以后不要求强制复制这个版本组合，只要
环境检查和 smoke test 通过即可。

## 4. 检查 GPU、T7 和协议

在仓库根目录执行：

```powershell
nvidia-smi

.\scripts\windows\check_environment.ps1 `
  -DataRoot "D:\ProjectData\Deepship\datasets\DeepShip"
```

检查脚本会以只读方式验证 Git 状态、PyTorch/CUDA、三份 manifest 哈希、20,000 段预算、
录音/船名组交集、T7 文件存在性、采样率和帧边界。任一检查失败都不能开始训练。

## 5. Smoke test

```powershell
.\scripts\windows\run_smoke.ps1 `
  -DataRoot "D:\ProjectData\Deepship\datasets\DeepShip" `
  -OutputRoot ".\runs\smoke_v1"
```

每个协议只运行 1 epoch 和少量 batch，用于验证 GPU 前向、反向、checkpoint、评估和输出链路。
smoke 结果不能作为质量指标。若命令行只显示 `Epoch 1/1`，可在对应 log 中确认完成状态；正式
脚本只按 epoch 输出，不打印每个 batch。

## 6. 一次性串行正式训练

运行三个协议的 seed 42/43：

```powershell
.\scripts\windows\run_formal.ps1 `
  -DataRoot "D:\ProjectData\Deepship\datasets\DeepShip" `
  -OutputRoot ".\runs\isolation_comparison_v1_epoch_output" `
  -Seeds 42,43 `
  -NumWorkers 0
```

脚本按 `segment_level → recording_disjoint → vessel_name_disjoint` 和 seed 自动串行执行，不需要
每个实验结束后重新输入命令。Windows 首次正式训练使用 `NumWorkers 0`，避免多进程 DataLoader
在移动硬盘或 PowerShell 环境中引入额外问题。

如果以后补原计划的 seed 44：

```powershell
.\scripts\windows\run_formal.ps1 `
  -DataRoot "D:\ProjectData\Deepship\datasets\DeepShip" `
  -OutputRoot ".\runs\isolation_comparison_v1_epoch_output" `
  -Seeds 44 `
  -NumWorkers 0
```

## 7. 中断恢复

每个 epoch 都保存 `models/deepship_macnna_last.pt`。只恢复中断的协议和 seed：

```powershell
.\scripts\windows\run_formal.ps1 `
  -DataRoot "D:\ProjectData\Deepship\datasets\DeepShip" `
  -OutputRoot ".\runs\isolation_comparison_v1_epoch_output" `
  -Seeds 42 `
  -Protocols "recording_disjoint" `
  -NumWorkers 0 `
  -Resume
```

存在 `reports/run_complete.json` 的 run 会拒绝恢复，防止意外追加训练。

## 8. 每个 run 的完整性

正式 run 至少包含：

```text
models/deepship_macnna_best.pt
models/deepship_macnna_last.pt
reports/environment.json
reports/frozen_split_manifest.json
reports/split_validation.json
reports/deepship_macnna_history.json
reports/deepship_macnna_run_config.json
reports/run_complete.json
metrics/segment_metrics.json
metrics/recording_metrics.json
metrics/vessel_metrics.json
predictions/test_segment_predictions.csv
predictions/test_recording_predictions.csv
predictions/test_vessel_predictions.csv
figures/deepship_macnna_training_curves.png
figures/segment_confusion_matrix.png
figures/recording_confusion_matrix.png
figures/vessel_confusion_matrix.png
```

只有 `run_complete.json` 状态为 `complete` 的正式 run 才进入汇总。本次六个 run 均满足上述
要求。

## 9. 回传和汇总

将整个
`C:\Users\shilongwang\Desktop\Deepship\runs\isolation_comparison_v1_epoch_output` 复制回本地，
不需要复制 T7 数据或缓存。本次复制后按协议/seed 整理在仓库根目录 `runs/`；该目录由 Git
忽略，不会把 checkpoint 推送到 GitHub。

严格汇总器按冻结配置要求完整的 3×3 矩阵：

```powershell
python scripts\eval\summarize_isolation_runs.py `
  --runs-root ".\runs\isolation_comparison_v1_epoch_output"
```

当前只有六个 run，因此现阶段的两 seed 结果直接从指标 JSON 计算，并写入 handbook 和文献
基准文档。补齐 seed 44 后，再使用严格汇总器生成三 seed 终稿。

## 10. 长任务注意事项

- 训练期间关闭自动睡眠和自动重启；
- 不要拔出 T7，也不要让其进入节能断连状态；
- 不要把 Git 仓库或 checkpoint 写在 T7；
- 每个 run 使用独立输出目录，脚本默认拒绝覆盖；
- 正式训练不要使用 `--allow-experiment-overrides`、`--max-train-batches` 或
  `--max-eval-batches`；
- 每个 epoch 的单行进度和最终指标同时写入日志，长时间无 batch 输出不代表程序卡死。
