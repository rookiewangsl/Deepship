from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random
import sys
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.shipsear import CLASS_NAMES, scan_shipsear

CLASS_DESCRIPTIONS = {
    "A": "small fishing / tug-like vessels",
    "B": "motorboat / sailboat-like vessels",
    "C": "passenger ferry",
    "D": "ocean liner / ro-ro vessels",
    "E": "background noise",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize representative ShipsEar mel spectrograms.")
    parser.add_argument("--data-root", default="ShipsEar")
    parser.add_argument("--output-path", default="outputs/figures/shipsear_mel_examples.png")
    parser.add_argument("--sample-rate", type=int, default=4000)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--f-min", type=float, default=20.0)
    parser.add_argument("--f-max", type=float, default=2000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples-per-class", type=int, default=1)
    parser.add_argument("--vmin", type=float, default=-35.0)
    parser.add_argument("--vmax", type=float, default=0.0)
    return parser


def select_examples(data_root: str, seed: int, examples_per_class: int) -> dict[str, list[str]]:
    records = scan_shipsear(data_root)
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[record.label_name].append(record.path)

    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for class_name in CLASS_NAMES:
        candidates = grouped[class_name][:]
        rng.shuffle(candidates)
        selected[class_name] = candidates[:examples_per_class]
    return selected


def plot_examples(args: argparse.Namespace) -> Path:
    chosen = select_examples(args.data_root, args.seed, args.examples_per_class)
    rows = len(CLASS_NAMES)
    cols = args.examples_per_class
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 3.2 * rows),
        squeeze=False,
    )

    for row_idx, class_name in enumerate(CLASS_NAMES):
        for col_idx, wav_path in enumerate(chosen[class_name]):
            ax = axes[row_idx][col_idx]
            waveform, sr = sf.read(wav_path, always_2d=False)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
            if sr != args.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, args.sample_rate)(waveform)
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=args.sample_rate,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.n_fft,
                n_mels=args.n_mels,
                f_min=args.f_min,
                f_max=args.f_max,
                power=2.0,
            )(waveform).squeeze(0).numpy()
            mel_db = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
            mel_db = mel_db - mel_db.max()
            duration = waveform.shape[-1] / args.sample_rate
            image = ax.imshow(
                mel_db,
                origin="lower",
                aspect="auto",
                cmap="jet",
                extent=[0, duration, args.f_min, args.f_max],
                vmin=args.vmin,
                vmax=args.vmax,
            )
            ax.set_title(
                f"Class {class_name}: {CLASS_DESCRIPTIONS[class_name]}\n{Path(wav_path).name}",
                fontsize=10,
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.text(
                0.015,
                0.93,
                f"{class_name}",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                color="white",
                va="top",
                ha="left",
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 4},
            )

    fig.suptitle("ShipsEar Representative Log-Mel Spectrograms by Class", fontsize=15)
    fig.tight_layout()
    cbar = fig.colorbar(image, ax=axes, format="%+2.0f dB", shrink=0.6)
    cbar.set_label("Log-Mel Energy (dB)")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    output_path = plot_examples(args)
    print(output_path)


if __name__ == "__main__":
    main()
