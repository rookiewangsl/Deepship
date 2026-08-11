# DeepShip 三协议 Windows GPU 训练指南

本指南用于在 Windows 电脑上读取挂载的 T7 数据，运行新 MA-CNN-A 网络的片段级、录音级和
船名组级三组对比实验。代码、协议和训练输出放 Windows 本机 SSD；T7 只读原始音频。

## 1. 目录约定

示例：

```text
D:\Code\Deepship                         # Git 仓库（本机 SSD）
D:\Runs\Deepship\isolation_comparison_v1 # 训练输出（本机 SSD）
E:\DeepShip\DeepShip                     # T7 上的原始数据
```

T7 的盘符可能不是 `E:`。每次接入后先在资源管理器或 PowerShell 中确认盘符。三个 manifest
只保存 `Cargo/...` 等相对路径，所以改变盘符不需要重新生成协议。

可在 PowerShell 中查找卷标和盘符：

```powershell
Get-Volume | Select-Object DriveLetter, FileSystemLabel, HealthStatus, SizeRemaining
```

确认后应检查例如 `E:\DeepShip\DeepShip\Cargo`、`Passenger`、`Tank` 和 `Tug` 四个目录确实
存在。不要根据示例默认 T7 一定是 `E:`。

## 2. 拉取代码

```powershell
Set-Location D:\Code
git clone git@github.com:rookiewangsl/Deepship.git
Set-Location D:\Code\Deepship
git rev-parse HEAD
git status --short
```

正式训练前记录 commit，并保持工作区干净。

## 3. 创建 Python 环境

建议使用 PyTorch 当前支持的 Python 3.11 或 3.12：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

随后访问 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)，选择：

- PyTorch Build：Stable；
- OS：Windows；
- Package：Pip；
- Language：Python；
- Compute Platform：与该电脑 NVIDIA 驱动兼容的 CUDA 版本。

执行页面生成的命令，并确保 `torch` 与 `torchaudio` 版本匹配。本项目不需要 torchvision。

## 4. 检查 GPU、T7 和三份协议

```powershell
nvidia-smi
.\scripts\windows\check_environment.ps1 -DataRoot "E:\DeepShip\DeepShip"
```

检查脚本会以只读方式验证：Git 工作区干净、当前 commit、PyTorch/CUDA、三份 manifest 哈希、
20,000 段预算、录音/船名组交集、T7 文件存在性、采样率和帧边界。任一检查失败都不能开始
训练。

## 5. 运行 smoke test

```powershell
.\scripts\windows\run_smoke.ps1 `
  -DataRoot "E:\DeepShip\DeepShip" `
  -OutputRoot "D:\Runs\Deepship\smoke_v1"
```

每个协议只运行 1 epoch、2 个训练 batch、2 个验证/测试 batch，用于验证 GPU 前向、反向、
checkpoint、评估和输出链路。smoke 结果不能作为正式指标。

## 6. 正式训练

先只跑三个协议的 seed 42：

```powershell
.\scripts\windows\run_formal.ps1 `
  -DataRoot "E:\DeepShip\DeepShip" `
  -OutputRoot "D:\Runs\Deepship\isolation_comparison_v1" `
  -Seeds 42 `
  -NumWorkers 0
```

三组均正常后补 seed 43 和 44：

```powershell
.\scripts\windows\run_formal.ps1 `
  -DataRoot "E:\DeepShip\DeepShip" `
  -OutputRoot "D:\Runs\Deepship\isolation_comparison_v1" `
  -Seeds 43,44 `
  -NumWorkers 0
```

Windows 首次正式训练使用 `NumWorkers 0`。链路稳定后可另做性能测试，再决定是否改为 2 或
4；三个正式协议应使用相同值。

## 7. 中断恢复

每个 epoch 都保存 `models/deepship_macnna_last.pt`。若某批运行中断，只恢复对应 seed，并保持
原输出目录：

```powershell
.\scripts\windows\run_formal.ps1 `
  -DataRoot "E:\DeepShip\DeepShip" `
  -OutputRoot "D:\Runs\Deepship\isolation_comparison_v1" `
  -Seeds 42 `
  -Protocols "recording_disjoint" `
  -NumWorkers 0 `
  -Resume
```

使用 `-Protocols` 只恢复中断的协议。存在 `run_complete.json` 的 run 会拒绝再次恢复，以免
意外追加训练。

## 8. 结果检查与回传

每个正式 run 应至少包含：

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
```

只有存在 `reports/run_complete.json` 且状态为 `complete` 的 run 才进入最终汇总。训练完成后将
整个 `D:\Runs\Deepship\isolation_comparison_v1` 复制回本地，不需要复制 T7 数据或缓存。

复制回本地后，在仓库根目录执行严格汇总：

```powershell
python scripts\eval\summarize_isolation_runs.py `
  --runs-root "D:\Runs\Deepship\isolation_comparison_v1"
```

脚本仅接受九个完整正式 run，并检查固定配置、模型参数量、manifest 哈希及 Git commit 一致性。

## 9. Windows 长任务注意事项

- 训练期间关闭自动睡眠和自动重启；
- 不要拔出 T7，也不要让其进入节能断连状态；
- 不要把 Git 仓库或 checkpoint 写在 T7；
- 每个 run 使用独立输出目录，脚本默认拒绝覆盖已有结果；
- 正式训练时不要使用 `--allow-experiment-overrides`、`--max-train-batches` 或
  `--max-eval-batches`。
