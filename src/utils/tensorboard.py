from __future__ import annotations

from pathlib import Path
import json
import os
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency
    sns = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None


def create_summary_writer(log_dir: str | Path):
    if SummaryWriter is None:
        print("TensorBoard is not available: install tensorboard to enable event logging.")
        return None
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(path))


def log_config(writer: Any, config: dict[str, object]) -> None:
    if writer is None:
        return
    writer.add_text("run/config", f"```json\n{json.dumps(config, indent=2)}\n```")


def add_spectrogram_batch(
    writer: Any,
    tag: str,
    batch: torch.Tensor,
    step: int = 0,
    max_items: int = 4,
    cmap: str = "magma",
) -> None:
    if writer is None:
        return
    specs = batch.detach().cpu()
    if specs.dim() == 4 and specs.size(1) == 1:
        specs = specs[:, 0]
    if specs.dim() != 3:
        return

    num_items = min(max_items, specs.size(0))
    fig, axes = plt.subplots(1, num_items, figsize=(4 * num_items, 3))
    if num_items == 1:
        axes = [axes]
    for idx in range(num_items):
        ax = axes[idx]
        image = specs[idx]
        ax.imshow(image.numpy(), origin="lower", aspect="auto", cmap=cmap)
        ax.set_title(f"sample_{idx}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Freq")
    fig.tight_layout()
    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)


def add_confusion_matrix(
    writer: Any,
    tag: str,
    conf_mat: list[list[int]],
    class_names: list[str],
    step: int,
) -> None:
    if writer is None:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    if sns is not None:
        sns.heatmap(
            conf_mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
    else:
        im = ax.imshow(conf_mat, cmap="Blues", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        for row_idx, row in enumerate(conf_mat):
            for col_idx, value in enumerate(row):
                ax.text(col_idx, row_idx, str(value), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(tag)
    fig.tight_layout()
    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)
