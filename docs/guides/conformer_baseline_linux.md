# DeepShip Wav2Vec2-Conformer 基线：Linux/RTX 4070 运行说明

最后更新：2026-08-28
状态：第二版 F0 已完成，F1b 因明确过拟合主动停止；S1 recording-balanced 动态裁剪已实现，
待服务器测试和 GPU 空闲后运行

## 1. 基线定义

首个基线对应计划中的 B3/F1：

```text
DeepShip frozen split manifest
→ 16 kHz 单通道原始波形
→ 以冻结 3 s manifest anchor 为中心读取 20 s 上下文
→ facebook/wav2vec2-conformer-rel-pos-large
→ Attentive Statistics Pooling
→ 四分类头
```

默认只训练最后 4 个 Conformer blocks、pooling 和分类头。脚本同时支持：

- `--random-init`：B2 从零训练对照；
- `--finetuning-mode frozen`：F0 冻结编码器；
- `--finetuning-mode last_n`：F1 部分解冻；
- `--finetuning-mode full`：F2 全量微调；
- `--clip-duration 3/10/20/30`：上下文消融。

干净基线显式关闭 checkpoint 内部默认的 latent SpecAugment 和 LayerDrop，避免在 B3 中提前混入
额外随机变量；只有后续增强消融才使用 `--apply-spec-augment` 或非零 `--layerdrop`。当 encoder
全冻结或仅解冻最后若干层时，冻结部分保持 evaluation mode，避免 dropout 改变冻结特征，也
减少不必要的反向图和显存占用。当前 12 GB RTX 4070 基线默认使用 BF16 并关闭 gradient
checkpointing；显式启用时使用非重入实现，以保证冻结前缀之后的可训练层仍能收到梯度。

训练默认每 100 个 batch 更新一次进度。可以用 `--log-interval N` 修改间隔，`N` 必须为正整数。
交互式终端在同一行覆盖显示 train/val 的累计 loss、accuracy、学习率、样本吞吐率和 CUDA 峰值
已分配显存，每个 epoch 完成后才固定一行摘要。使用 `2>&1 | tee` 时，动态进度直接写到控制终端，
日志文件只保存 epoch 摘要；若在没有控制终端的后台环境运行，则自动退化为周期性普通日志。

所选官方预训练 checkpoint 是在 960 h、16 kHz LibriSpeech 上自监督预训练的 24 层、hidden
size 1024、16 头 relative-position large 版本。当前分类系统共 619,353,477 个参数，`last-4`
模式有 101,701,381 个可训练参数。选择它是为了建立公开可复现的强通用迁移基线，不因为它是
水声专用模型；后续必须用 scratch、冻结和不同解冻深度分离预训练与参数规模的贡献。

该版本还不是 recording-level MIL。S0 保持冻结 manifest 的 14,000/4,000/2,000 个 anchor；S1
在训练集改为 class→recording-balanced 的 14,000 次动态裁剪，validation/test 仍保持冻结 anchor。

## 2. 当前服务器目录

```text
/home/slwang/workspace/Deepship/        # Git 仓库
/home/slwang/.venvs/deepship/           # Python 3.11 环境
/home/slwang/deepship/
  datasets/DeepShip/
  datasets/PORTIA/
  pretrained/huggingface/
  runs/conformer_baseline_v1/
  env.sh
```

代码和小型配置在 Git 仓库中；音频、Hugging Face 权重和 run 输出位于挂载为 `/home` 的 7.3 TB
服务器数据盘，不使用 70 GB 根分区保存大文件。

## 3. 环境准备

在 Linux 服务器按服务器 CUDA/驱动版本使用 PyTorch 官方安装选择器安装匹配的 `torch` 和
`torchaudio`，随后安装项目依赖：

```bash
python3.11 -m venv /home/slwang/.venvs/deepship
source /home/slwang/.venvs/deepship/bin/activate
python -m pip install --upgrade pip
# 先按 https://pytorch.org/get-started/locally/ 安装匹配服务器 CUDA 的 torch/torchaudio
python -m pip install -r requirements-conformer.txt
```

设置大文件缓存和数据路径：

```bash
source /home/slwang/deepship/env.sh
# 等价的核心变量：
export DEEPSHIP_DATA_ROOT=/home/slwang/deepship/datasets/DeepShip
export HF_HOME=/home/slwang/deepship/pretrained/huggingface
```

预训练权重约 2.5 GB。首次运行由 Transformers 下载；正式训练前应记录最终缓存 revision 和
模型文件哈希。训练代码会把 Hugging Face 解析出的 commit 写入
`reports/model_report.json`；后续复现实验用 `--pretrained-revision <commit>` 固定它。

## 4. 先做环境预检、静态与单元测试

环境和数据预检不会下载模型或启动训练：

```bash
python scripts/check_conformer_environment.py \
  --data-root "$DEEPSHIP_DATA_ROOT" \
  --split-manifest protocols/isolation_comparison_v1/vessel_name_disjoint/split_manifest.json \
  --output-root /home/slwang/deepship/runs/conformer_baseline_v1
```

预检默认要求模型缓存至少剩余 5 GiB、run 输出盘至少剩余 20 GiB。正式进行多 seed 前应按每个
run 同时保留 best/last checkpoint 和优化器状态另行预留空间，不能把 20 GiB 当成全部实验的
容量估计。

只有输出 JSON 的 `status` 为 `passed`，才继续单元测试：

```bash
python -m unittest \
  tests.test_conformer_baseline_config \
  tests.test_deepship_waveform \
  tests.test_wav2vec2_conformer
```

然后验证原始 frozen protocol：

```bash
python scripts/prepare/validate_deepship_protocols.py \
  --data-root "$DEEPSHIP_DATA_ROOT" \
  --protocol-root protocols/isolation_comparison_v1 \
  --no-write-reports
```

## 5. 服务器 smoke run

先使用 vessel-name-disjoint、冻结 encoder、3 s 和极少 batch 检查数据、权重、显存、反向和输出：

```bash
python scripts/train/train_deepship_conformer.py \
  --data-root "$DEEPSHIP_DATA_ROOT" \
  --split-manifest protocols/isolation_comparison_v1/vessel_name_disjoint/split_manifest.json \
  --protocol-name vessel_name_disjoint \
  --pretrained-revision 1afaab48b41d924fbbcae05d8c5d88836c4a5719 \
  --output-root /home/slwang/deepship/runs/conformer_baseline_v1/smoke_frozen_3s_seed42 \
  --clip-duration 3 \
  --finetuning-mode frozen \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --num-workers 0
```

smoke 通过后再测试 20 s、最后 4 层解冻：

```bash
python scripts/train/train_deepship_conformer.py \
  --data-root "$DEEPSHIP_DATA_ROOT" \
  --split-manifest protocols/isolation_comparison_v1/vessel_name_disjoint/split_manifest.json \
  --protocol-name vessel_name_disjoint \
  --pretrained-revision 1afaab48b41d924fbbcae05d8c5d88836c4a5719 \
  --output-root /home/slwang/deepship/runs/conformer_baseline_v1/smoke_last4_20s_bf16_no_gc_seed42_v2 \
  --clip-duration 20 \
  --finetuning-mode last_n \
  --train-last-n-layers 4 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --precision bf16 \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --num-workers 0
```

2026-08-28 的服务器验证发现，旧的 FP16＋重入 gradient checkpointing 组合会让最后 4 层没有
梯度；FP16 首批梯度也可能非有限。当前代码因此默认使用 BF16、batch size 1、梯度累积 8，并
关闭 checkpointing。两批次 20 s/last-4 smoke 的 `nvidia-smi` 采样峰值约 6.7 GiB，最后 4 层、
pooling 和分类头均有有限梯度。第二版正式训练把 last-4 encoder 峰值学习率降为 `5e-6`，head
峰值学习率降为 `1e-4`，并改为逐 optimizer step 更新的 5% warmup＋cosine decay。

只有显存不足时才显式加入 `--gradient-checkpointing`；该选项使用非重入 checkpointing，启用后
必须重新通过梯度回归检查和 smoke。4070 可能是 8 GB、12 GB 或其他显存配置，不能只凭型号
假设 20 s/last-4 一定可运行；若第二个 smoke 显存不足，按
`20 s/last-2 → 10 s/last-4 → 10 s/last-2` 的顺序排查，并把可运行配置与原计划的差异写入报告。

## 6. 首轮正式运行顺序

### 每个 epoch 的日志

每轮开始时立即显示 `batch=0`，随后每 100 batch 在同一个终端行上覆盖更新。下面三行表示同一
物理行在不同时刻的内容，实际不会纵向累积：

```text
Epoch 1/30 | train | batch=0/14000 (0.0%) | avg_loss=-- | avg_acc=-- | lr=head:1.00e-05 | samples_per_sec=-- | gpu_peak=2.34GiB
Epoch 1/30 | train | batch=100/14000 (0.7%) | avg_loss=1.4321 | avg_acc=0.2900 | lr=head:1.04e-05 | samples_per_sec=2.35 | gpu_peak=6.71GiB
...
Epoch 1/30 | train | batch=14000/14000 (100.0%) | avg_loss=0.8421 | avg_acc=0.6812 | lr=head:7.00e-05 | samples_per_sec=2.38 | gpu_peak=6.72GiB
```

训练结束后，同一物理行切换到验证进度：

```text
Epoch 1/30 | val | batch=100/4000 (2.5%) | avg_loss=1.1034 | avg_acc=0.5600 | lr=head:7.00e-05 | samples_per_sec=3.41 | gpu_peak=2.94GiB
...
Epoch 1/30 | val | batch=4000/4000 (100.0%) | avg_loss=0.9912 | avg_acc=0.5905 | lr=head:7.00e-05 | samples_per_sec=3.45 | gpu_peak=2.95GiB
```

保存 history 和 best/last checkpoint 后，清除动态行并固定本轮唯一的最终输出：

```text
Epoch 1/30 | done | train_loss=0.8421 | train_acc=0.6812 | val_loss=0.9912 | val_acc=0.5905 | val_recording_f1=0.6012 | val_vessel_f1=0.5840 | select=vessel_macro_f1:0.5840 | best_select=0.5840 | time=02:08:31
```

上述数字仅用于展示格式，不是实际实验结果。学习率在每次 optimizer update 后变化，终端显示
当前值；F0 的 encoder 完全冻结，因此动态行显示 `lr=head:...`，F1b 会显示
`lr=enc:...,head:...`。vessel-name-disjoint 以 validation vessel macro-F1 选择 checkpoint；如果
连续 5 epoch 不提升，最终摘要追加 `early_stop=true` 和 `best_epoch`。窄终端会截断动态行而不换行，
完整 epoch 摘要和结构化 history 不受影响。

1. 第二版 F0 已完成，最佳 validation vessel macro-F1 为 0.4726；
2. F1b 最佳值为 0.4650，并在后续出现明显过拟合，已停止；不运行 last-8；
3. 下一项运行 F0-S1，检验固定 anchor 冗余与 recording 暴露偏置；
4. F0-S1 提高至少 1 pp 且 recording macro-F1 不退化时，再运行 S1＋last-2；
5. S2 仅由 S1 收益或残余 vessel 暴露偏置触发；
6. 若 S1 无效，转入 3/10/20 s 上下文和 scratch/预训练控制；
7. 只有最佳候选进入 42/43/44/45/46 多 seed。

不要先运行所有因素的笛卡尔积，也不要在看 test 后修改聚合规则。

## 7. S1 smoke 与正式运行

新管线默认不评估 DeepShip test；最终方案完全冻结前不要添加
`--evaluate-test-on-completion`。为避免多行命令粘贴错位，优先使用已冻结参数的启动脚本：

```bash
bash scripts/train/run_conformer_f0_s1_seed42.sh smoke
```

该脚本的 smoke 模式等价于：

```bash
python scripts/train/train_deepship_conformer.py \
  --data-root "$DEEPSHIP_DATA_ROOT" \
  --split-manifest protocols/isolation_comparison_v1/vessel_name_disjoint/split_manifest.json \
  --protocol-name vessel_name_disjoint \
  --pretrained-revision 1afaab48b41d924fbbcae05d8c5d88836c4a5719 \
  --output-root /home/slwang/deepship/runs/conformer_sampling_v1/smoke_f0_s1_20s_seed42 \
  --clip-duration 20 \
  --finetuning-mode frozen \
  --training-sampling recording_balanced_dynamic \
  --train-samples-per-epoch 14000 \
  --batch-size 1 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --precision bf16 \
  --epochs 1 \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.005 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --num-workers 0
```

smoke 通过且 GPU 没有其他项目占用后运行：

```bash
bash scripts/train/run_conformer_f0_s1_seed42.sh formal
```

正式模式使用新目录 `formal_f0_s1_recording_dynamic_20s_seed42`、`--epochs 8` 和
`--num-workers 4`。它会额外保存 `reports/training_sampling_exposure.json`，其中含每轮
recording/vessel 暴露、重复率、吞吐和 DataLoader 等待占比。

## 8. 当前未实现范围

- ONC 水声自监督适配（B4）；
- S2 vessel-balanced 动态采样；
- recording-level MIL（当前已使用 ASP，但 loss 仍是单窗口级）；
- 真实背景混合（B6）；
- 门控频谱支路（B7）；
- PORTIA-4 和连续 Oceanship/ONC-4 外测读取器。

这些功能不是固定依次累加：只有前一阶段的验证或外测结果指向明确缺口时才加入。尤其是 ONC
自监督只在 B3 内部有效但外部泛化不足时启动，双前端只在线谱相关错误持续存在时启动。
