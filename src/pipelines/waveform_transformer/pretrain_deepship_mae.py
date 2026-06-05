from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.models.waveform_transformer import STFTMAEPretrainer
from src.pipelines.mel_ml.train_shipsear_cnn import get_default_device, set_seed


@dataclass
class PretrainConfig:
    precomputed_root: str = "outputs/precomputed/deepship_stft"
    output_root: str = "outputs"
    seed: int = 42
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    scheduler_tmax: int = 50
    num_workers: int = 0
    mask_ratio: float = 0.75
    time_mask_param: int = 30
    freq_mask_param: int = 8
    noise_std_min: float = 0.02
    noise_std_max: float = 0.10
    color_noise_std_min: float = 0.02
    color_noise_std_max: float = 0.08
    stripe_prob: float = 0.3
    random_gain_db_min: float = -1.0
    random_gain_db_max: float = 0.5
    patch_size_freq: int = 32
    patch_size_time: int = 8
    embed_dim: int = 96
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    decoder_embed_dim: int = 64
    decoder_layers: int = 2
    decoder_heads: int = 4
    device: str = get_default_device()


class PrecomputedSTFTMAEDataset(Dataset):
    def __init__(
        self,
        bundle_path: str | Path,
        time_mask_param: int = 30,
        freq_mask_param: int = 8,
        noise_std_min: float = 0.02,
        noise_std_max: float = 0.10,
        color_noise_std_min: float = 0.02,
        color_noise_std_max: float = 0.08,
        stripe_prob: float = 0.3,
        random_gain_db_min: float = -1.0,
        random_gain_db_max: float = 0.5,
    ) -> None:
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        self.features: torch.Tensor = bundle["features"].float()
        self.time_mask = torch.nn.Sequential(
            torch.nn.Identity()
        )
        self.time_mask_aug = torch.nn.ModuleList(
            [
                torch.nn.Identity(),
            ]
        )
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.color_noise_std_min = color_noise_std_min
        self.color_noise_std_max = color_noise_std_max
        self.stripe_prob = stripe_prob
        self.random_gain_db_min = random_gain_db_min
        self.random_gain_db_max = random_gain_db_max
        self.time_mask_op = torch.nn.Identity()
        self.freq_mask_op = torch.nn.Identity()
        self._torchaudio_time_mask = None
        self._torchaudio_freq_mask = None

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean = self.features[index].clone()
        noisy = self._augment(clean.clone())
        return noisy, clean

    def _augment(self, spec: torch.Tensor) -> torch.Tensor:
        if self._torchaudio_time_mask is None:
            self._init_mask_ops()
        gain_db = float(torch.empty(1).uniform_(self.random_gain_db_min, self.random_gain_db_max).item())
        spec = spec * (10 ** (gain_db / 20.0))

        if torch.rand(1).item() < 0.7:
            std = float(torch.empty(1).uniform_(self.noise_std_min, self.noise_std_max).item())
            spec = spec + torch.randn_like(spec) * std

        if torch.rand(1).item() < 0.5:
            std = float(torch.empty(1).uniform_(self.color_noise_std_min, self.color_noise_std_max).item())
            freq_weights = torch.linspace(1.0, 0.3, steps=spec.size(-2), dtype=spec.dtype).view(1, -1, 1)
            spec = spec + torch.randn_like(spec) * freq_weights * std

        if torch.rand(1).item() < self.stripe_prob:
            n_stripes = int(torch.randint(1, 4, (1,)).item())
            for _ in range(n_stripes):
                start = int(torch.randint(0, spec.size(-2), (1,)).item())
                width = int(torch.randint(1, 3, (1,)).item())
                boost = float(torch.empty(1).uniform_(0.05, 0.15).item())
                spec[:, start : min(start + width, spec.size(-2)), :] += boost

        if torch.rand(1).item() < 0.5:
            spec = self._torchaudio_time_mask(spec)
        if torch.rand(1).item() < 0.5:
            spec = self._torchaudio_freq_mask(spec)
        if torch.rand(1).item() < 0.3:
            spec = self._torchaudio_time_mask(spec)
        if torch.rand(1).item() < 0.3:
            spec = self._torchaudio_freq_mask(spec)

        mean = spec.mean()
        std = spec.std().clamp_min(1e-6)
        return (spec - mean) / std

    def _init_mask_ops(self) -> None:
        import torchaudio

        self._torchaudio_time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=self.time_mask_param
        )
        self._torchaudio_freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=self.freq_mask_param
        )


def pretrain(config: PretrainConfig) -> Path:
    set_seed(config.seed)
    precomputed_root = Path(config.precomputed_root)
    train_path = precomputed_root / "train.pt"
    if not train_path.exists():
        raise FileNotFoundError(f"Precomputed STFT bundle not found: {train_path}")

    output_root = Path(config.output_root)
    models_dir = output_root / "models"
    figures_dir = output_root / "figures"
    reports_dir = output_root / "reports"
    for directory in [models_dir, figures_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    dataset = PrecomputedSTFTMAEDataset(
        bundle_path=train_path,
        time_mask_param=config.time_mask_param,
        freq_mask_param=config.freq_mask_param,
        noise_std_min=config.noise_std_min,
        noise_std_max=config.noise_std_max,
        color_noise_std_min=config.color_noise_std_min,
        color_noise_std_max=config.color_noise_std_max,
        stripe_prob=config.stripe_prob,
        random_gain_db_min=config.random_gain_db_min,
        random_gain_db_max=config.random_gain_db_max,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    sample = dataset.features[0]
    input_size = (int(sample.shape[-2]), int(sample.shape[-1]))
    model = STFTMAEPretrainer(
        input_size=input_size,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.scheduler_tmax or config.epochs),
    )

    history = {"train_loss": []}
    best_loss = math.inf
    best_path = models_dir / "deepship_stft_mae_pretrained.pt"

    print(f"Starting MAE pretraining on device={config.device} for {config.epochs} epochs...")
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        progress = tqdm(dataloader, desc=f"MAE Epoch {epoch:03d}/{config.epochs:03d}", dynamic_ncols=True)
        for noisy, clean in progress:
            noisy = noisy.to(config.device)
            clean = clean.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model(noisy, clean, mask_ratio=config.mask_ratio)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(1, total_batches)
        history["train_loss"].append(avg_loss)
        print(f"Epoch {epoch:03d}/{config.epochs:03d} train_loss={avg_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            state = {
                "encoder_state_dict": {
                    k: v
                    for k, v in model.state_dict().items()
                    if not k.startswith("decoder_") and k != "mask_token"
                },
                "config": asdict(config),
                "input_size": input_size,
                "best_loss": best_loss,
                "model_family": "deepship_stft_mae",
            }
            torch.save(state, best_path)
            print(f"New best MAE checkpoint saved with loss={best_loss:.4f}")

    (reports_dir / "deepship_stft_mae_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    curve_path = figures_dir / "deepship_stft_mae_training_curves.png"
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DeepShip STFT-MAE Pretraining Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved MAE checkpoint to {best_path}")
    return best_path
