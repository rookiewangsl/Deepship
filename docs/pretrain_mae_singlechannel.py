# -*- coding: utf-8 -*-
"""
单通道 STFT 时频图 MAE 预训练 (参考 pretrain_mae_deep.py 架构)

wav → 重采样 → STFT(50-500Hz) → 128x128 → MAE 重建
含域噪声增强 (白噪声/有色噪声/条纹干扰) + SpecAugment

用法:
  python pretrain_mae_singlechannel.py
  python pretrain_mae_singlechannel.py --highpass 30
"""

import os
import sys
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)


class Config:
    wav_dir = os.path.join(os.path.dirname(__file__), "train_dataset_wav")
    sample_rate = 1000
    duration = 5.0
    n_fft = 256
    hop_length = 32
    freq_min = 50
    freq_max = 500

    img_size = 128
    patch_size = 8       # 与 pretrain_mae_deep 一致
    embed_dim = 128
    num_heads = 8
    encoder_layers = 6
    decoder_embed_dim = 64
    decoder_num_heads = 4
    decoder_layers = 3
    mlp_ratio = 2.0
    mask_ratio = 0.75

    batch_size = 32
    epochs = 100
    lr = 1e-3
    weight_decay = 0.05
    highpass_freq = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def n_patches(self):
        return (self.img_size // self.patch_size) ** 2


# ==================== 数据集 ====================
class WavPretrainDataset(Dataset):
    def __init__(self, wav_dir, config):
        self.config = config
        self.files = []
        for root, _, fnames in os.walk(wav_dir):
            for f in fnames:
                if f.endswith(".wav"):
                    self.files.append(os.path.join(root, f))
        print(f"预训练样本: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        noisy, clean = self._load(self.files[idx])
        return torch.FloatTensor(noisy).unsqueeze(0), torch.FloatTensor(clean).unsqueeze(0)

    def _load(self, fp):
        try:
            sr, audio = wavfile.read(fp)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = audio.astype(np.float32)
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            if sr != self.config.sample_rate:
                audio = signal.resample(audio,
                    int(len(audio) * self.config.sample_rate / sr)).astype(np.float32)
            target_len = int(self.config.sample_rate * self.config.duration)
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]

            # 高通滤波
            if self.config.highpass_freq > 0:
                sos = signal.butter(3, self.config.highpass_freq, btype="highpass",
                                    fs=self.config.sample_rate, output="sos")
                audio = signal.sosfiltfilt(sos, audio.astype(np.float64),
                    padlen=min(500, len(audio) - 1)).astype(np.float32)

            audio *= 10 ** random.uniform(-1, 0.5)
            clean = self._stft(audio, augment=False)
            noisy = self._stft(audio, augment=True)
            return noisy, clean
        except Exception:
            z = np.zeros((self.config.img_size, self.config.img_size), dtype=np.float32)
            return z, z

    def _stft(self, audio, augment=False):
        cfg = self.config
        f, t, Sxx = signal.spectrogram(audio, fs=cfg.sample_rate,
            nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop_length)
        # 频率裁剪
        mask = (f >= cfg.freq_min) & (f <= cfg.freq_max)
        Sxx = Sxx[mask, :]
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        vmin, vmax = Sxx_db.min(), Sxx_db.max()
        if vmax - vmin > 1e-10:
            img = (Sxx_db - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(Sxx_db)
        img = zoom(img, (cfg.img_size / img.shape[0],
                         cfg.img_size / img.shape[1]), order=1)
        if augment:
            img = self._spec_augment(img)
            img = self._domain_noise(img)
        return img.astype(np.float32)

    def _domain_noise(self, spec):
        h, w = spec.shape
        if random.random() < 0.7:
            spec = spec + np.random.normal(0, random.uniform(0.02, 0.1), spec.shape)
        if random.random() < 0.5:
            fw = np.linspace(1.0, 0.3, h).reshape(-1, 1)
            spec = spec + np.random.normal(0, random.uniform(0.02, 0.08), spec.shape) * fw
        if random.random() < 0.3:
            for _ in range(random.randint(1, 3)):
                p = random.randint(0, h - 1)
                w2 = random.randint(1, 2)
                spec[p:min(p + w2, h), :] += random.uniform(0.05, 0.15)
        return np.clip(spec, 0, 1)

    def _spec_augment(self, spec):
        h, w = spec.shape
        if random.random() < 0.5:
            tw = random.randint(2, max(3, w // 10))
            ts = random.randint(0, w - tw)
            spec[:, ts:ts + tw] = spec.mean()
        if random.random() < 0.5:
            fh = random.randint(2, max(3, h // 10))
            fs = random.randint(0, h - fh)
            spec[fs:fs + fh, :] = spec.mean()
        if random.random() < 0.3:
            tw = random.randint(2, max(3, w // 15))
            ts = random.randint(0, w - tw)
            spec[:, ts:ts + tw] = spec.mean()
        if random.random() < 0.3:
            fh = random.randint(2, max(3, h // 15))
            fs = random.randint(0, h - fh)
            spec[fs:fs + fh, :] = spec.mean()
        return spec


# ==================== MAE 模型 ====================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0):
        super().__init__()
        h = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class MAESingleCh(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        n_p = config.n_patches
        self.patch_embed = PatchEmbedding(config.img_size, config.patch_size, config.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_p, config.embed_dim))
        self.encoder_blocks = nn.ModuleList([
            Block(config.embed_dim, config.num_heads, config.mlp_ratio)
            for _ in range(config.encoder_layers)])
        self.encoder_norm = nn.LayerNorm(config.embed_dim)
        self.decoder_embed = nn.Linear(config.embed_dim, config.decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, n_p, config.decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(config.decoder_embed_dim, config.decoder_num_heads, config.mlp_ratio)
            for _ in range(config.decoder_layers)])
        self.decoder_norm = nn.LayerNorm(config.decoder_embed_dim)
        self.decoder_pred = nn.Linear(config.decoder_embed_dim, config.patch_size ** 2)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        x_masked = torch.gather(x, 1, ids_shuffle[:, :keep].unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    def forward(self, noisy, clean=None, mask_ratio=0.75):
        x = self.patch_embed(noisy) + self.pos_embed
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        x = self.decoder_embed(x)
        B, Nv, D = x.shape
        Nt = ids_restore.shape[1]
        tokens = torch.cat([x, self.mask_token.expand(B, Nt - Nv, -1)], dim=1)
        x = torch.gather(tokens, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        pred = self.decoder_pred(self.decoder_norm(x))
        target = self.patchify(clean if clean is not None else noisy)
        loss = ((pred - target) ** 2).mean(dim=-1)
        return (loss * mask).sum() / mask.sum(), pred, mask

    def patchify(self, imgs):
        p = self.config.patch_size
        h = w = self.config.img_size // p
        x = imgs.reshape(imgs.shape[0], 1, h, p, w, p)
        return x.permute(0, 2, 4, 3, 5, 1).reshape(imgs.shape[0], h * w, p * p)


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="单通道 STFT MAE 预训练")
    parser.add_argument("--highpass", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    args = parser.parse_args()

    config = Config()
    config.highpass_freq = args.highpass
    config.epochs = args.epochs

    print("=" * 60)
    print("单通道 STFT MAE 预训练 (wav)")
    print(f"  数据: {config.wav_dir}")
    print(f"  STFT: n_fft={config.n_fft}, hop={config.hop_length}")
    print(f"  频率: {config.freq_min}-{config.freq_max} Hz")
    print(f"  图像: {config.img_size}x{config.img_size}")
    print(f"  ViT: patch={config.patch_size}, dim={config.embed_dim}, "
          f"heads={config.num_heads}, layers={config.encoder_layers}")
    print(f"  设备: {config.device}")
    if config.highpass_freq > 0:
        print(f"  高通: {config.highpass_freq} Hz")
    print("=" * 60)

    dataset = WavPretrainDataset(config.wav_dir, config)
    if len(dataset) == 0:
        print("没有数据")
        return
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    model = MAESingleCh(config).to(config.device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for noisy, clean in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
            noisy, clean = noisy.to(config.device), clean.to(config.device)
            optimizer.zero_grad()
            loss, _, _ = model(noisy, clean, mask_ratio=config.mask_ratio)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{config.epochs} Loss={avg:.4f}")

    save_path = os.path.join(os.path.dirname(__file__),
                             "mae_singlechannel_pretrained.pth")
    state = {k: v for k, v in model.state_dict().items()
             if "decoder" not in k and "mask_token" not in k}
    torch.save({
        "encoder_state_dict": state,
        "config": {
            "img_size": config.img_size, "patch_size": config.patch_size,
            "embed_dim": config.embed_dim, "num_heads": config.num_heads,
            "encoder_layers": config.encoder_layers,
            "n_fft": config.n_fft, "hop_length": config.hop_length,
            "freq_min": config.freq_min, "freq_max": config.freq_max,
            "highpass_freq": config.highpass_freq,
            "sample_rate": config.sample_rate,
            "model_family": "mae_singlechannel",
        },
    }, save_path)
    print(f"已保存: {save_path}")


if __name__ == "__main__":
    main()
