from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data.deepship import (
    DeepShipWaveformDataset,
    build_segment_records,
    save_split_manifest,
    scan_deepship,
    stratified_split,
    summarize_records,
    summarize_segments,
)
from src.models.waveform_transformer import STFTMAEPretrainer
from src.pipelines.mel_ml.train_shipsear_cnn import get_default_device, set_seed
from src.utils.optim import EpochWarmupCosineScheduler
from src.utils.pathing import resolve_path
from src.utils.tensorboard import add_spectrogram_batch, create_summary_writer, log_config


@dataclass
class PretrainConfig:
    data_root: str = "DeepShip"
    output_root: str = "outputs"
    sample_rate: int = 3000
    clip_duration: float = 5.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    min_lr: float = 1e-5
    num_workers: int = 0
    mask_ratio: float = 0.75
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    highpass_freq: float = 50.0
    freq_min: float = 50.0
    freq_max: float = 1500.0
    img_h: int = 128
    img_w: int = 128
    time_mask_param: int = 12
    freq_mask_param: int = 12
    noise_std_min: float = 0.02
    noise_std_max: float = 0.10
    color_noise_std_min: float = 0.02
    color_noise_std_max: float = 0.08
    stripe_prob: float = 0.3
    patch_size_freq: int = 8
    patch_size_time: int = 8
    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    decoder_embed_dim: int = 64
    decoder_layers: int = 3
    decoder_heads: int = 4
    device: str = get_default_device()


class DynamicSTFTMAEDataset(Dataset):
    def __init__(
        self,
        segments,
        sample_rate: int = 3000,
        clip_duration: float = 5.0,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        highpass_freq: float = 50.0,
        freq_min: float = 50.0,
        freq_max: float = 1500.0,
        img_h: int = 128,
        img_w: int = 128,
        time_mask_param: int = 12,
        freq_mask_param: int = 12,
        noise_std_min: float = 0.02,
        noise_std_max: float = 0.10,
        color_noise_std_min: float = 0.02,
        color_noise_std_max: float = 0.08,
        stripe_prob: float = 0.3,
    ) -> None:
        self.segments = segments
        self.sample_rate = sample_rate
        self.clip_samples = int(sample_rate * clip_duration)
        self.highpass_freq = highpass_freq
        self.img_h = img_h
        self.img_w = img_w
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.color_noise_std_min = color_noise_std_min
        self.color_noise_std_max = color_noise_std_max
        self.stripe_prob = stripe_prob
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2.0,
            center=True,
            window_fn=torch.hann_window,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self._torchaudio_time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=time_mask_param
        )
        self._torchaudio_freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=freq_mask_param
        )
        freqs = torch.linspace(0, sample_rate / 2, steps=(n_fft // 2 + 1))
        self.freq_mask_tensor = (freqs >= freq_min) & (freqs <= freq_max)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        segment = self.segments[index]
        waveform, sr = DeepShipWaveformDataset._load_segment(segment)
        clean = self._build_clean_spec(waveform, sr)
        noisy = self._augment_spec(clean.clone())
        return noisy, clean

    def _build_clean_spec(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.sample_rate:
            waveform = self._resample(waveform, sr)
        waveform = self._fix_length(waveform)
        if self.highpass_freq > 0:
            waveform = AF.highpass_biquad(waveform, self.sample_rate, self.highpass_freq)
        waveform = self._standardize_waveform(waveform)
        spec = self.spectrogram(waveform)
        spec = spec[:, self.freq_mask_tensor, :]
        spec = self.amplitude_to_db(spec)
        spec = torch.nn.functional.interpolate(
            spec.unsqueeze(0),
            size=(self.img_h, self.img_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return self._standardize_spec(spec)

    def _augment_spec(self, spec: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.7:
            std = float(torch.empty(1).uniform_(self.noise_std_min, self.noise_std_max).item())
            spec = spec + torch.randn_like(spec) * std

        if torch.rand(1).item() < 0.5:
            std = float(
                torch.empty(1).uniform_(
                    self.color_noise_std_min,
                    self.color_noise_std_max,
                ).item()
            )
            freq_weights = torch.linspace(
                1.0,
                0.3,
                steps=spec.size(-2),
                dtype=spec.dtype,
            ).view(1, -1, 1)
            spec = spec + torch.randn_like(spec) * freq_weights * std

        if torch.rand(1).item() < self.stripe_prob:
            n_stripes = int(torch.randint(1, 4, (1,)).item())
            for _ in range(n_stripes):
                start = int(torch.randint(0, spec.size(-2), (1,)).item())
                width = int(torch.randint(1, 3, (1,)).item())
                boost = float(torch.empty(1).uniform_(0.05, 0.15).item())
                spec[:, start:min(start + width, spec.size(-2)), :] += boost

        if torch.rand(1).item() < 0.5:
            spec = self._torchaudio_time_mask(spec)
        if torch.rand(1).item() < 0.5:
            spec = self._torchaudio_freq_mask(spec)
        if torch.rand(1).item() < 0.3:
            spec = self._torchaudio_time_mask(spec)
        if torch.rand(1).item() < 0.3:
            spec = self._torchaudio_freq_mask(spec)

        return self._standardize_spec(spec)

    def _resample(self, waveform: torch.Tensor, src_sr: int) -> torch.Tensor:
        if src_sr not in self.resamplers:
            self.resamplers[src_sr] = torchaudio.transforms.Resample(
                orig_freq=src_sr,
                new_freq=self.sample_rate,
            )
        return self.resamplers[src_sr](waveform)

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        num_samples = waveform.size(-1)
        if num_samples > self.clip_samples:
            waveform = waveform[..., :self.clip_samples]
        elif num_samples < self.clip_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.clip_samples - num_samples))
        return waveform

    @staticmethod
    def _standardize_waveform(waveform: torch.Tensor) -> torch.Tensor:
        mean = waveform.mean(dim=-1, keepdim=True)
        std = waveform.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (waveform - mean) / std

    @staticmethod
    def _standardize_spec(spec: torch.Tensor) -> torch.Tensor:
        mean = spec.mean()
        std = spec.std().clamp_min(1e-6)
        return (spec - mean) / std


def pretrain(config: PretrainConfig) -> str:
    set_seed(config.seed)
    records = scan_deepship(config.data_root)
    train_records, val_records, test_records = stratified_split(
        records=records,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    train_segments = build_segment_records(train_records, clip_duration=config.clip_duration)
    val_segments = build_segment_records(val_records, clip_duration=config.clip_duration)

    output_root = resolve_path(config.output_root)
    models_dir = output_root / "models"
    figures_dir = output_root / "figures"
    reports_dir = output_root / "reports"
    tensorboard_dir = output_root / "tensorboard"
    for directory in [models_dir, figures_dir, reports_dir, tensorboard_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    writer = create_summary_writer(tensorboard_dir)
    log_config(writer, asdict(config))

    save_split_manifest(reports_dir, train_records, val_records, test_records)
    split_stats = {
        "full_recordings": summarize_records(records),
        "train_recordings": summarize_records(train_records),
        "val_recordings": summarize_records(val_records),
        "test_recordings": summarize_records(test_records),
        "train_segments": summarize_segments(train_segments),
        "val_segments": summarize_segments(val_segments),
        "test_segments": summarize_segments(
            build_segment_records(test_records, clip_duration=config.clip_duration)
        ),
    }
    (reports_dir / "deepship_stft_mae_split_stats.json").write_text(
        json.dumps(split_stats, indent=2),
        encoding="utf-8",
    )

    train_dataset = DynamicSTFTMAEDataset(
        segments=train_segments,
        sample_rate=config.sample_rate,
        clip_duration=config.clip_duration,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        highpass_freq=config.highpass_freq,
        freq_min=config.freq_min,
        freq_max=config.freq_max,
        img_h=config.img_h,
        img_w=config.img_w,
        time_mask_param=config.time_mask_param,
        freq_mask_param=config.freq_mask_param,
        noise_std_min=config.noise_std_min,
        noise_std_max=config.noise_std_max,
        color_noise_std_min=config.color_noise_std_min,
        color_noise_std_max=config.color_noise_std_max,
        stripe_prob=config.stripe_prob,
    )
    val_dataset = DynamicSTFTMAEDataset(
        segments=val_segments,
        sample_rate=config.sample_rate,
        clip_duration=config.clip_duration,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        highpass_freq=config.highpass_freq,
        freq_min=config.freq_min,
        freq_max=config.freq_max,
        img_h=config.img_h,
        img_w=config.img_w,
        time_mask_param=0,
        freq_mask_param=0,
        noise_std_min=0.0,
        noise_std_max=0.0,
        color_noise_std_min=0.0,
        color_noise_std_max=0.0,
        stripe_prob=0.0,
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=str(config.device).startswith("cuda"),
        persistent_workers=(config.num_workers > 0),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=str(config.device).startswith("cuda"),
        persistent_workers=(config.num_workers > 0),
    )
    sample_noisy, sample_clean = next(iter(dataloader))
    add_spectrogram_batch(writer, "samples/noisy", sample_noisy)
    add_spectrogram_batch(writer, "samples/clean", sample_clean)

    model = STFTMAEPretrainer(
        input_size=(config.img_h, config.img_w),
        patch_size=(config.patch_size_freq, config.patch_size_time),
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_layers=config.decoder_layers,
        decoder_heads=config.decoder_heads,
    ).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: STFT MAE, {n_params:,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = str(config.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()
    scheduler = EpochWarmupCosineScheduler(
        optimizer,
        total_epochs=config.epochs,
        warmup_epochs=config.warmup_epochs,
        warmup_start_factor=config.warmup_start_factor,
        min_lr=config.min_lr,
    )

    history = {"train_loss": [], "val_loss": []}
    best_loss = math.inf
    best_path = models_dir / "deepship_stft_mae_best.pt"

    print(
        f"Starting MAE pretraining from wav-derived STFT on device={config.device} "
        f"for {config.epochs} epochs..."
    )
    for epoch in range(1, config.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        model.train()
        total_loss = 0.0
        total_batches = 0
        progress = tqdm(
            dataloader,
            desc=f"MAE Epoch {epoch:03d}/{config.epochs:03d}",
            dynamic_ncols=True,
        )
        for noisy, clean in progress:
            noisy = noisy.to(config.device, non_blocking=use_amp)
            clean = clean.to(config.device, non_blocking=use_amp)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                loss, _, _ = model(noisy, clean, mask_ratio=config.mask_ratio)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, total_batches)
        history["train_loss"].append(avg_loss)

        model.eval()
        val_total_loss = 0.0
        val_total_batches = 0
        with torch.no_grad():
            for noisy, clean in val_dataloader:
                noisy = noisy.to(config.device, non_blocking=use_amp)
                clean = clean.to(config.device, non_blocking=use_amp)
                with amp_ctx:
                    val_loss, _, _ = model(noisy, clean, mask_ratio=config.mask_ratio)
                val_total_loss += float(val_loss.item())
                val_total_batches += 1
        avg_val_loss = val_total_loss / max(1, val_total_batches)
        history["val_loss"].append(avg_val_loss)
        if writer is not None:
            writer.add_scalar("loss/train", avg_loss, epoch)
            writer.add_scalar("loss/val", avg_val_loss, epoch)
            writer.add_scalar("lr", current_lr, epoch)
        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} "
            f"train_loss={avg_loss:.4f} val_loss={avg_val_loss:.4f} lr={current_lr:.6f}"
        )
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            state = {
                "encoder_state_dict": {
                    k: v
                    for k, v in model.state_dict().items()
                    if not k.startswith("decoder_") and k != "mask_token"
                },
                "config": asdict(config),
                "input_size": (config.img_h, config.img_w),
                "best_loss": best_loss,
                "model_family": "deepship_stft_mae",
            }
            torch.save(state, best_path)
            if writer is not None:
                writer.add_scalar("loss/best", best_loss, epoch)
            print(f"New best MAE checkpoint saved with loss={best_loss:.4f}")

    (reports_dir / "deepship_stft_mae_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    curve_path = figures_dir / "deepship_stft_mae_training_curves.png"
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DeepShip STFT-MAE Pretraining Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=200)
    plt.close()
    if writer is not None:
        writer.flush()
        writer.close()
    print(f"Saved MAE checkpoint to {best_path}")
    return str(best_path)
