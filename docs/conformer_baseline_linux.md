# DeepShip Wav2Vec2-Conformer 基线：Linux/RTX 4070 运行说明

最后更新：2026-08-26
状态：代码已实现并通过静态检查；尚未在安装 PyTorch 的 Linux 服务器执行前向或训练

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
减少不必要的反向图和显存占用；最后若干可训练层仍启用 gradient checkpointing。

所选官方预训练 checkpoint 是 24 层、hidden size 1024 的 large 版本，比计划文档中用于起步估算
的 12 层、80M～120M 方案更大。选择它是因为存在可直接复用的官方预训练权重；不把它与计划
中的中型 scratch Conformer 混称为同一模型。后续架构因果线仍需加入参数更受控的 C0～C3。

该版本还不是 recording-level MIL；它保持当前冻结 manifest 的 14,000/4,000/2,000 个 anchor
预算，先建立可对照的 raw-waveform 预训练基线。MIL 和 recording-balanced sampling 属于后续 B5。

## 2. 服务器目录建议

```text
/data/Deepship/
  code/
  datasets/DeepShip/
  pretrained/huggingface/
  runs/conformer_baseline_v1/
```

代码和小型配置可以在 Git 仓库中，音频、Hugging Face 权重和 run 输出应在服务器数据盘。

## 3. 环境准备

在 Linux 服务器按服务器 CUDA/驱动版本使用 PyTorch 官方安装选择器安装匹配的 `torch` 和
`torchaudio`，随后安装项目依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 先按 https://pytorch.org/get-started/locally/ 安装匹配服务器 CUDA 的 torch/torchaudio
python -m pip install -r requirements-conformer.txt
```

设置大文件缓存和数据路径：

```bash
export DEEPSHIP_DATA_ROOT=/data/Deepship/datasets/DeepShip
export HF_HOME=/data/Deepship/pretrained/huggingface
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
  --output-root /data/Deepship/runs/conformer_baseline_v1
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
  --output-root /data/Deepship/runs/conformer_baseline_v1/smoke_frozen_3s_seed42 \
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
  --output-root /data/Deepship/runs/conformer_baseline_v1/smoke_last4_20s_seed42 \
  --clip-duration 20 \
  --finetuning-mode last_n \
  --train-last-n-layers 4 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --num-workers 0
```

RTX 4070 默认使用 FP16、batch size 1、梯度累积 8 和 gradient checkpointing。是否需要调整最后
解冻层数、上下文或 batch，只根据 smoke 的实测峰值显存决定。4070 可能是 8 GB、12 GB 或其他
显存配置，不能只凭型号假设 20 s/last-4 一定可运行；若第二个 smoke 显存不足，按
`20 s/last-2 → 10 s/last-4 → 10 s/last-2` 的顺序排查，并把可运行配置与原计划的差异写入报告。

## 6. 首轮正式运行顺序

1. F0：20 s，encoder frozen，seed 42；
2. F1：20 s，last 4 blocks，seed 42；
3. T：F1 下比较 3/10/20 s，seed 42；
4. 选定上下文后运行 recording-disjoint 和 vessel-name-disjoint；
5. 只有最佳候选进入 42/43/44/45/46 多 seed。

不要先运行所有因素的笛卡尔积，也不要在看 test 后修改聚合规则。

## 7. 当前未实现范围

- ONC 水声自监督适配（B4）；
- recording-level MIL/ASP 训练采样（B5；当前已使用 ASP，但 loss 仍是 anchor-level）；
- 真实背景混合（B6）；
- 门控频谱支路（B7）；
- PORTIA-4 和连续 Oceanship/ONC-4 外测读取器。

这些功能应在首个 B3 基线前向、显存和训练链路验证后依次加入。
