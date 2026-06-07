from __future__ import annotations

import math


class EpochWarmupCosineScheduler:
    """Epoch-based linear warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer,
        *,
        total_epochs: int,
        warmup_epochs: int = 5,
        warmup_start_factor: float = 0.1,
        min_lr: float = 1e-5,
    ) -> None:
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        if not (0.0 < warmup_start_factor <= 1.0):
            raise ValueError("warmup_start_factor must be in (0, 1]")
        if min_lr < 0.0:
            raise ValueError("min_lr must be non-negative")

        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = max(0, min(warmup_epochs, total_epochs))
        self.warmup_start_factor = warmup_start_factor
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_epoch = 1
        self._set_lrs(self.current_epoch)

    def _lr_for_epoch(self, base_lr: float, epoch: int) -> float:
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            if self.warmup_epochs == 1:
                factor = 1.0
            else:
                progress = (epoch - 1) / (self.warmup_epochs - 1)
                factor = self.warmup_start_factor + (
                    1.0 - self.warmup_start_factor
                ) * progress
            return base_lr * factor

        if self.total_epochs <= self.warmup_epochs:
            return base_lr

        cosine_span = self.total_epochs - self.warmup_epochs
        if cosine_span == 1:
            progress = 1.0
        else:
            progress = (epoch - self.warmup_epochs - 1) / (cosine_span - 1)
            progress = min(max(progress, 0.0), 1.0)
        return self.min_lr + (base_lr - self.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def _set_lrs(self, epoch: int) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self._lr_for_epoch(base_lr, epoch)

    def step(self) -> None:
        self.current_epoch = min(self.current_epoch + 1, self.total_epochs)
        self._set_lrs(self.current_epoch)

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]
