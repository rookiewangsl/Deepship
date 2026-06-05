# -*- coding: utf-8 -*-
"""
单通道时频图 ViT 分类训练

直接用 wav 文件 → STFT 时频图 → ViT 6分类
用于多通道粗筛后的精细识别

数据: train_dataset_wav/ 下按类别分文件夹的 wav 文件
模型: 6层 Transformer + 全连接分类头

用法:
  python train_singlechannel_spec.py
  python train_singlechannel_spec.py --highpass 30
"""

import os
import sys
import random
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import wavfile
from scipy.signal import spectrogram, butter, sosfiltfilt, resample
from scipy.ndimage import zoom
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)


# ==================== 配置 ====================
class Config:
    wav_dir = os.path.join(os.path.dirname(__file__), "train_dataset_wav")
    classes = ["A_Passenger", "B_Motorboat", "C_Cargo",
               "D_Seismic", "E_CW", "F_Background",
               "G_Tug", "H_Tanker", "I_FlowNoise"]
    num_classes = len(classes)

    target_sr = 1000
    target_duration = 5.0

    # STFT (频率分辨率 1000/n_fft Hz；n_fft=256 → 3.9Hz，便于区分船型谐波线谱)
    n_fft = 256
    hop_length = 32
    freq_min = 50
    freq_max = 500

    # 图像
    img_h = 128
    img_w = 128

    # ViT (与预训练一致)
    patch_size = 8          # 与 pretrain_mae_deep 一致
    embed_dim = 128
    num_heads = 8
    num_layers = 6
    mlp_ratio = 2.0
    dropout = 0.1

    # 训练
    batch_size = 32
    epochs = 100
    lr = 3e-4
    weight_decay = 1e-4
    highpass_freq = 0
    label_smoothing = 0.1  # 标签平滑，减少过拟合

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def n_patches(self):
        return (self.img_h // self.patch_size) * (self.img_w // self.patch_size)


# ==================== 数据加载 ====================
def load_wav_as_stft(file_path, config, augment=False):
    """wav → 重采样 → 可选高通 → STFT → 归一化 → (1, img_h, img_w)"""
    try:
        sr, audio = wavfile.read(file_path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        # 重采样到目标采样率
        if sr != config.target_sr:
            n_dst = int(len(audio) * config.target_sr / sr)
            audio = resample(audio, n_dst).astype(np.float32)

        # 截取/填充到目标时长
        target_len = int(config.target_sr * config.target_duration)
        if len(audio) > target_len:
            if augment:
                start = random.randint(0, len(audio) - target_len)
            else:
                start = 0
            audio = audio[start:start + target_len]
        elif len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))

        # 高通滤波
        if config.highpass_freq > 0:
            sos = butter(3, config.highpass_freq, btype="highpass",
                         fs=config.target_sr, output="sos")
            audio = sosfiltfilt(sos, audio.astype(np.float64),
                                padlen=min(500, len(audio) - 1)).astype(np.float32)

        # 数据增强
        if augment:
            audio *= 10 ** random.uniform(-0.3, 0.3)  # 随机增益
            if random.random() < 0.3:
                audio += np.random.normal(0, 0.01, len(audio)).astype(np.float32)
            if random.random() < 0.2:  # 时间偏移
                shift = random.randint(-500, 500)
                audio = np.roll(audio, shift)

        # STFT (与预训练一致)
        f, t, Sxx = spectrogram(audio, fs=config.target_sr,
            nperseg=config.n_fft, noverlap=config.n_fft - config.hop_length)
        # 频率裁剪
        freq_mask = (f >= config.freq_min) & (f <= config.freq_max)
        Sxx = Sxx[freq_mask, :]
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # 归一化
        vmin, vmax = Sxx_db.min(), Sxx_db.max()
        if vmax - vmin > 1e-10:
            img = (Sxx_db - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(Sxx_db)

        # 缩放
        img = zoom(img, (config.img_h / img.shape[0],
                         config.img_w / img.shape[1]), order=1)

        # 时频遮挡增强
        if augment:
            h, w = img.shape
            if random.random() < 0.4:  # 频率遮挡
                fh = random.randint(3, max(4, h // 8))
                fs_start = random.randint(0, h - fh)
                img[fs_start:fs_start + fh, :] = 0
            if random.random() < 0.4:  # 时间遮挡
                tw = random.randint(3, max(4, w // 8))
                ts = random.randint(0, w - tw)
                img[:, ts:ts + tw] = 0

        return img[np.newaxis].astype(np.float32)

    except Exception as e:
        print(f"Error: {file_path}: {e}")
        return np.zeros((1, config.img_h, config.img_w), dtype=np.float32)


class WavDataset(Dataset):
    def __init__(self, file_paths, labels, config, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.config = config
        self.augment = augment
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        img = load_wav_as_stft(self.file_paths[idx], self.config, self.augment)
        return torch.FloatTensor(img), self.labels[idx]


# ==================== ViT 模型 ====================
class PatchEmbedding(nn.Module):
    def __init__(self, img_h, img_w, patch_size, embed_dim):
        super().__init__()
        self.n_patches = (img_h // patch_size) * (img_w // patch_size)
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.drop((q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1))
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=3.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, dim), nn.Dropout(dropout))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class SingleChViT(nn.Module):
    """6层 Transformer + 全连接分类头"""
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            config.img_h, config.img_w, config.patch_size, config.embed_dim)
        n_p = config.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_p + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            Block(config.embed_dim, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.embed_dim)
        # 两层分类头（比单层更强）
        self.head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 2, config.num_classes))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0]))


# ==================== 主函数 ====================
def load_dataset(config):
    fps, labels = [], []
    for ci, cls in enumerate(config.classes):
        d = os.path.join(config.wav_dir, cls)
        if not os.path.exists(d):
            continue
        cnt = 0
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(".wav"):
                    fps.append(os.path.join(root, f))
                    labels.append(ci)
                    cnt += 1
        print(f"  {cls}: {cnt}")
    print(f"  总计: {len(fps)}")
    return fps, labels


def main():
    parser = argparse.ArgumentParser(description="单通道时频图 ViT 训练 (wav)")
    parser.add_argument("--highpass", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    args = parser.parse_args()

    config = Config()
    config.highpass_freq = args.highpass
    config.epochs = args.epochs

    output_dir = os.path.dirname(__file__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("单通道时频图 ViT 分类训练 (wav → STFT)")
    print(f"  数据: {config.wav_dir}")
    print(f"  STFT: n_fft={config.n_fft}, hop={config.hop_length}, freq={config.freq_min}-{config.freq_max}Hz")
    print(f"  图像: {config.img_h}x{config.img_w}")
    print(f"  ViT: {config.num_layers}层, dim={config.embed_dim}, heads={config.num_heads}")
    if config.highpass_freq > 0:
        print(f"  高通: {config.highpass_freq} Hz")
    print(f"  设备: {config.device}")
    print("=" * 60)

    print("\n加载数据...")
    fps, labels = load_dataset(config)
    if not fps:
        print("没有数据")
        return

    train_p, val_p, train_l, val_l = train_test_split(
        fps, labels, test_size=0.2, random_state=42, stratify=labels)
    print(f"训练: {len(train_p)}, 验证: {len(val_p)}")

    train_ds = WavDataset(train_p, train_l, config, augment=True)
    val_ds = WavDataset(val_p, val_l, config, augment=False)

    counter = Counter(train_l)
    weights = [1.0 / counter[l] for l in train_l]
    sampler = WeightedRandomSampler(weights, len(train_l), replacement=True)
    loader_workers = min(4, os.cpu_count() or 1)
    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": loader_workers,
        "pin_memory": pin_memory,
    }
    if loader_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = SingleChViT(config).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")

    # 加载 MAE 预训练权重（如果有）
    pt_path = os.path.join(output_dir, "mae_singlechannel_pretrained.pth")
    if pt_path and os.path.exists(pt_path):
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        mae_state = ckpt["encoder_state_dict"]
        ms = model.state_dict()
        loaded = 0
        for mk in mae_state:
            if mk.startswith("patch_embed."):
                ck = mk
            elif mk == "pos_embed":
                ms["pos_embed"][:, 1:, :] = mae_state[mk]
                loaded += 1
                continue
            elif mk.startswith("encoder_blocks."):
                ck = mk.replace("encoder_blocks.", "blocks.")
            elif mk.startswith("encoder_norm."):
                ck = mk.replace("encoder_norm.", "norm.")
            else:
                continue
            if ck in ms and mae_state[mk].shape == ms[ck].shape:
                ms[ck] = mae_state[mk]
                loaded += 1
        model.load_state_dict(ms)
        print(f"预训练权重: {loaded} 组")

    # 标签平滑交叉熵
    cw = torch.zeros(config.num_classes)
    for ci in range(config.num_classes):
        cw[ci] = len(train_l) / (config.num_classes * max(counter.get(ci, 1), 1))
    criterion = nn.CrossEntropyLoss(weight=cw.to(config.device),
                                    label_smoothing=config.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc = 0.0
    hp_tag = f"_hp{int(config.highpass_freq)}" if config.highpass_freq > 0 else "_nohp"
    save_path = os.path.join(output_dir, f"model_singlechannel_{timestamp}{hp_tag}.pth")

    amp_ctx = torch.cuda.amp.autocast if config.device.type == "cuda" else nullcontext

    for epoch in range(config.epochs):
        model.train()
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            imgs = imgs.to(config.device, non_blocking=True)
            lbls = lbls.to(config.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                logits = model(imgs)
                loss = criterion(logits, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(config.device, non_blocking=True)
                lbls = lbls.to(config.device, non_blocking=True)
                with amp_ctx():
                    logits = model(imgs)
                _, preds = logits.max(1)
                correct += (preds == lbls).sum().item()
                total += imgs.size(0)
        val_acc = correct / total
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{config.epochs} Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "img_h": config.img_h, "img_w": config.img_w,
                    "patch_size": config.patch_size,
                    "embed_dim": config.embed_dim, "num_heads": config.num_heads,
                    "num_layers": config.num_layers, "num_classes": config.num_classes,
                    "mlp_ratio": config.mlp_ratio,
                    "dropout": config.dropout,
                    "classes": config.classes,
                    "n_fft": config.n_fft,
                    "hop_length": config.hop_length,
                    "freq_min": config.freq_min,
                    "freq_max": config.freq_max,
                    "highpass_freq": config.highpass_freq,
                    "target_sr": config.target_sr,
                    "model_family": "singlechannel_spec",
                },
                "val_acc": best_acc, "epoch": epoch + 1,
            }, save_path)

    print(f"\n最佳: {best_acc:.4f}, 模型: {os.path.basename(save_path)}")
    print("重新加载最佳模型进行最终评估...")
    best_ckpt = torch.load(save_path, map_location=config.device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    final_preds, final_labels = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(config.device, non_blocking=True)
            lbls = lbls.to(config.device, non_blocking=True)
            with amp_ctx():
                logits = model(imgs)
            _, preds = logits.max(1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(lbls.cpu().numpy())

    precision, recall, f1, support = precision_recall_fscore_support(
        final_labels, final_preds, labels=list(range(config.num_classes)), zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    print(f"Macro-F1={macro_f1:.4f}")
    for cls_name, rec in zip(config.classes, recall):
        print(f"  Recall[{cls_name}]={rec:.4f}")

    report = classification_report(final_labels, final_preds,
                                   target_names=config.classes, digits=4)
    print(report)

    cm = confusion_matrix(final_labels, final_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.classes, yticklabels=config.classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"SingleCh STFT ViT Acc={best_ckpt['val_acc']:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_singlechannel.png"), dpi=150)
    plt.close()
    print(f"混淆矩阵已保存，最佳 Epoch={best_ckpt['epoch']}, 最佳 Acc={best_ckpt['val_acc']:.4f}")


if __name__ == "__main__":
    main()
