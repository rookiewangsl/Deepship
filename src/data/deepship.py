from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
from typing import Iterable

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


CLASS_NAMES = ["Cargo", "Passenger", "Tank", "Tug"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class AudioRecord:
    path: str
    class_name: str
    label_index: int
    sample_rate: int
    num_frames: int
    duration_seconds: float


@dataclass(frozen=True)
class SegmentRecord:
    path: str
    class_name: str
    label_index: int
    start_frame: int
    num_frames: int
    sample_rate: int
    segment_index: int
    total_segments: int


def scan_deepship(root_dir: str | Path) -> list[AudioRecord]:
    root = Path(root_dir)
    records: list[AudioRecord] = []
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for wav_path in sorted(class_dir.rglob("*.wav")):
            info = sf.info(str(wav_path))
            records.append(
                AudioRecord(
                    path=str(wav_path),
                    class_name=class_name,
                    label_index=CLASS_TO_INDEX[class_name],
                    sample_rate=info.samplerate,
                    num_frames=info.frames,
                    duration_seconds=info.frames / info.samplerate,
                )
            )
    if not records:
        raise FileNotFoundError(f"No wav files found under {root}")
    return records


def summarize_records(records: Iterable[AudioRecord]) -> dict[str, object]:
    records = list(records)
    class_counts = Counter(record.class_name for record in records)
    total_duration_seconds = sum(record.duration_seconds for record in records)
    avg_duration_seconds = total_duration_seconds / len(records) if records else 0.0
    return {
        "num_recordings": len(records),
        "class_counts": {name: class_counts.get(name, 0) for name in CLASS_NAMES},
        "total_duration_seconds": total_duration_seconds,
        "avg_duration_seconds": avg_duration_seconds,
    }


def summarize_segments(segments: Iterable[SegmentRecord]) -> dict[str, object]:
    segments = list(segments)
    class_counts = Counter(segment.class_name for segment in segments)
    return {
        "num_segments": len(segments),
        "class_counts": {name: class_counts.get(name, 0) for name in CLASS_NAMES},
    }


def stratified_split(
    records: list[AudioRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[AudioRecord], list[AudioRecord], list[AudioRecord]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    grouped: dict[int, list[AudioRecord]] = {}
    for record in records:
        grouped.setdefault(record.label_index, []).append(record)

    rng = random.Random(seed)
    train_records: list[AudioRecord] = []
    val_records: list[AudioRecord] = []
    test_records: list[AudioRecord] = []

    for label_index in sorted(grouped):
        items = grouped[label_index][:]
        rng.shuffle(items)
        n_total = len(items)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        if n_train <= 0:
            n_train = 1
        if n_val <= 0:
            n_val = 1
        if n_train + n_val >= n_total:
            n_val = max(1, n_total - n_train - 1)
        n_test = n_total - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1

        train_records.extend(items[:n_train])
        val_records.extend(items[n_train : n_train + n_val])
        test_records.extend(items[n_train + n_val :])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)
    return train_records, val_records, test_records


def save_split_manifest(
    split_dir: str | Path,
    train_records: list[AudioRecord],
    val_records: list[AudioRecord],
    test_records: list[AudioRecord],
) -> None:
    split_path = Path(split_dir)
    split_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": [record.__dict__ for record in train_records],
        "val": [record.__dict__ for record in val_records],
        "test": [record.__dict__ for record in test_records],
    }
    (split_path / "deepship_split_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_segment_records(
    records: Iterable[AudioRecord],
    clip_duration: float = 5.0,
) -> list[SegmentRecord]:
    segments: list[SegmentRecord] = []
    for record in records:
        clip_frames = int(round(record.sample_rate * clip_duration))
        total_segments = max(1, math.ceil(record.num_frames / clip_frames))
        for segment_index in range(total_segments):
            start_frame = segment_index * clip_frames
            remaining = max(0, record.num_frames - start_frame)
            num_frames = min(clip_frames, remaining) if remaining > 0 else 0
            segments.append(
                SegmentRecord(
                    path=record.path,
                    class_name=record.class_name,
                    label_index=record.label_index,
                    start_frame=start_frame,
                    num_frames=num_frames,
                    sample_rate=record.sample_rate,
                    segment_index=segment_index,
                    total_segments=total_segments,
                )
            )
    return segments


class DeepShipMelDataset(Dataset):
    def __init__(
        self,
        segments: list[SegmentRecord],
        sample_rate: int = 4000,
        clip_duration: float = 5.0,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
        n_mels: int = 64,
        f_min: float = 20.0,
        f_max: float = 2000.0,
        augment: bool = False,
        cache_features: bool = False,
        time_shift_frames: int = 8,
        time_mask_param: int = 12,
        freq_mask_param: int = 8,
    ) -> None:
        self.segments = segments
        self.sample_rate = sample_rate
        self.clip_samples = int(sample_rate * clip_duration)
        self.augment = augment
        self.cache_features = cache_features
        self.time_shift_frames = time_shift_frames
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            center=True,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.cached_examples: list[tuple[torch.Tensor, int]] | None = None
        if self.cache_features and not self.augment:
            self.cached_examples = [self._build_example(segment) for segment in self.segments]

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self.cached_examples is not None:
            mel_spec, label_index = self.cached_examples[index]
            return mel_spec.clone(), label_index
        return self._build_example(self.segments[index])

    def _build_example(self, segment: SegmentRecord) -> tuple[torch.Tensor, int]:
        waveform, sr = self._load_segment(segment)
        if sr != self.sample_rate:
            waveform = self._resample(waveform, sr)
        waveform = self._fix_length(waveform)
        if self.augment:
            waveform = self._augment_waveform(waveform)
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        mel_spec = self._normalize(mel_spec)
        if self.augment:
            mel_spec = self._augment_mel(mel_spec)
        return mel_spec, segment.label_index

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
            waveform = waveform[..., : self.clip_samples]
        elif num_samples < self.clip_samples:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, self.clip_samples - num_samples),
            )
        return waveform

    def _augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        shift = int(torch.randint(low=-400, high=401, size=(1,)).item())
        waveform = torch.roll(waveform, shifts=shift, dims=-1)
        gain = float(torch.empty(1).uniform_(0.85, 1.15).item())
        waveform = waveform * gain
        waveform = waveform.clamp(-1.0, 1.0)
        return waveform

    @staticmethod
    def _normalize(mel_spec: torch.Tensor) -> torch.Tensor:
        mean = mel_spec.mean()
        std = mel_spec.std().clamp_min(1e-6)
        return (mel_spec - mean) / std

    def _augment_mel(self, mel_spec: torch.Tensor) -> torch.Tensor:
        if self.time_shift_frames > 0:
            shift = int(torch.randint(
                low=-self.time_shift_frames,
                high=self.time_shift_frames + 1,
                size=(1,),
            ).item())
            mel_spec = torch.roll(mel_spec, shifts=shift, dims=-1)
        mel_spec = self.time_mask(mel_spec)
        mel_spec = self.freq_mask(mel_spec)
        return mel_spec

    @staticmethod
    def _load_segment(segment: SegmentRecord) -> tuple[torch.Tensor, int]:
        audio, sample_rate = sf.read(
            segment.path,
            start=segment.start_frame,
            frames=segment.num_frames if segment.num_frames > 0 else -1,
            dtype="float32",
            always_2d=True,
        )
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1, keepdims=True)
        waveform = torch.from_numpy(audio[:, 0]).unsqueeze(0)
        return waveform, sample_rate
