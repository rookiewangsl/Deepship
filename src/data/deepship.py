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

from src.utils.pathing import resolve_path


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
    root = resolve_path(root_dir)
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
    items = list(records)
    class_counts = Counter(item.class_name for item in items)
    total_duration_seconds = sum(item.duration_seconds for item in items)
    return {
        "num_recordings": len(items),
        "class_counts": {name: class_counts.get(name, 0) for name in CLASS_NAMES},
        "total_duration_seconds": total_duration_seconds,
        "avg_duration_seconds": total_duration_seconds / len(items) if items else 0.0,
    }


def summarize_segments(segments: Iterable[SegmentRecord]) -> dict[str, object]:
    items = list(segments)
    class_counts = Counter(item.class_name for item in items)
    return {
        "num_segments": len(items),
        "class_counts": {name: class_counts.get(name, 0) for name in CLASS_NAMES},
    }


def build_segment_records(
    records: Iterable[AudioRecord],
    clip_duration: float,
) -> list[SegmentRecord]:
    segments: list[SegmentRecord] = []
    for record in records:
        clip_frames = int(round(record.sample_rate * clip_duration))
        total_segments = record.num_frames // clip_frames
        for segment_index in range(total_segments):
            start_frame = segment_index * clip_frames
            segments.append(
                SegmentRecord(
                    path=record.path,
                    class_name=record.class_name,
                    label_index=record.label_index,
                    start_frame=start_frame,
                    num_frames=clip_frames,
                    sample_rate=record.sample_rate,
                    segment_index=segment_index,
                    total_segments=total_segments,
                )
            )
    return segments


def build_paper_split(
    records: list[AudioRecord],
    *,
    clip_duration: float = 3.0,
    samples_per_class: int = 5000,
    train_per_class: int = 3500,
    val_per_class: int = 1000,
    test_per_class: int = 500,
    seed: int = 42,
) -> tuple[dict[str, list[SegmentRecord]], dict[str, object]]:
    expected_total = train_per_class + val_per_class + test_per_class
    if samples_per_class != expected_total:
        raise ValueError("samples_per_class must equal train_per_class + val_per_class + test_per_class")

    all_segments = build_segment_records(records, clip_duration=clip_duration)
    grouped: dict[str, list[SegmentRecord]] = {name: [] for name in CLASS_NAMES}
    for segment in all_segments:
        grouped[segment.class_name].append(segment)

    rng = random.Random(seed)
    split_segments = {"train": [], "val": [], "test": []}
    split_stats: dict[str, object] = {
        "recordings": summarize_records(records),
        "all_segments": summarize_segments(all_segments),
        "paper_protocol": {
            "clip_duration": clip_duration,
            "samples_per_class": samples_per_class,
            "train_per_class": train_per_class,
            "val_per_class": val_per_class,
            "test_per_class": test_per_class,
            "sampling_rule": (
                "recording-level split first, then segment-level sampling from "
                "non-overlapping 3-second segments within each split"
            ),
        },
        "available_segments_per_class": {},
        "selected_segments_per_class": {},
        "selected_recordings_per_class": {},
    }

    for class_name in CLASS_NAMES:
        items = grouped[class_name][:]
        rng.shuffle(items)
        if len(items) < samples_per_class:
            raise ValueError(
                f"Class {class_name} only has {len(items)} full segments, fewer than requested {samples_per_class}"
            )
        split_stats["available_segments_per_class"][class_name] = len(items)

        by_recording: dict[str, list[SegmentRecord]] = {}
        for segment in items:
            by_recording.setdefault(segment.path, []).append(segment)

        recordings = list(by_recording.items())
        rng.shuffle(recordings)

        test_records = _select_recording_subset(recordings, test_per_class)
        remaining_records = [item for item in recordings if item[0] not in {path for path, _ in test_records}]

        val_records = _select_recording_subset(remaining_records, val_per_class)
        remaining_records = [item for item in remaining_records if item[0] not in {path for path, _ in val_records}]

        train_records = _select_recording_subset(remaining_records, train_per_class)

        selected_recordings = {
            "train": train_records,
            "val": val_records,
            "test": test_records,
        }
        selected_segments = {
            split_name: _sample_segments_from_recordings(recording_items, target_count, rng)
            for split_name, recording_items, target_count in [
                ("train", train_records, train_per_class),
                ("val", val_records, val_per_class),
                ("test", test_records, test_per_class),
            ]
        }
        for split_name, segments in selected_segments.items():
            split_segments[split_name].extend(segments)

        split_stats["selected_segments_per_class"][class_name] = {
            split_name: len(segments)
            for split_name, segments in selected_segments.items()
        }
        split_stats["selected_recordings_per_class"][class_name] = {
            split_name: {
                "num_recordings": len(recording_items),
                "available_segments": sum(len(segments) for _, segments in recording_items),
            }
            for split_name, recording_items in selected_recordings.items()
        }

    for split_name in split_segments:
        rng.shuffle(split_segments[split_name])

    split_stats["train_segments"] = summarize_segments(split_segments["train"])
    split_stats["val_segments"] = summarize_segments(split_segments["val"])
    split_stats["test_segments"] = summarize_segments(split_segments["test"])
    split_stats["sampled_recording_overlap"] = _summarize_recording_overlap(split_segments)
    return split_segments, split_stats


def _select_recording_subset(
    recording_items: list[tuple[str, list[SegmentRecord]]],
    target_segments: int,
) -> list[tuple[str, list[SegmentRecord]]]:
    totals = [len(segments) for _, segments in recording_items]
    predecessor: dict[int, tuple[int, int] | None] = {0: None}

    for index, total in enumerate(totals):
        updates: dict[int, tuple[int, int]] = {}
        for current_sum in list(predecessor):
            next_sum = current_sum + total
            if next_sum not in predecessor and next_sum not in updates:
                updates[next_sum] = (current_sum, index)
        predecessor.update(updates)

    candidate_sum = next((value for value in sorted(predecessor) if value >= target_segments), None)
    if candidate_sum is None:
        raise ValueError(
            f"Unable to allocate recordings with at least {target_segments} segments "
            f"(available: {sum(totals)})"
        )

    chosen_indices: set[int] = set()
    current_sum = candidate_sum
    while current_sum != 0:
        previous_sum, index = predecessor[current_sum]
        chosen_indices.add(index)
        current_sum = previous_sum

    return [item for index, item in enumerate(recording_items) if index in chosen_indices]


def _sample_segments_from_recordings(
    recording_items: list[tuple[str, list[SegmentRecord]]],
    target_segments: int,
    rng: random.Random,
) -> list[SegmentRecord]:
    pool = [segment for _, segments in recording_items for segment in segments]
    if len(pool) < target_segments:
        raise ValueError(
            f"Recording subset only contains {len(pool)} segments, fewer than requested {target_segments}"
        )
    rng.shuffle(pool)
    return pool[:target_segments]


def _summarize_recording_overlap(split_segments: dict[str, list[SegmentRecord]]) -> dict[str, object]:
    by_split = {
        split_name: Counter(segment.path for segment in segments)
        for split_name, segments in split_segments.items()
    }
    all_paths = set().union(*[set(counter) for counter in by_split.values()])
    overlapping_paths = {
        path: sorted(split_name for split_name, counter in by_split.items() if path in counter)
        for path in all_paths
        if sum(path in counter for counter in by_split.values()) > 1
    }
    return {
        "num_overlapping_recordings": len(overlapping_paths),
        "examples": [
            {"path": path, "splits": splits}
            for path, splits in list(sorted(overlapping_paths.items()))[:10]
        ],
    }


def save_segment_split_manifest(
    output_dir: str | Path,
    split_segments: dict[str, list[SegmentRecord]],
    split_stats: dict[str, object],
) -> None:
    path = resolve_path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        split: [segment.__dict__ for segment in segments]
        for split, segments in split_segments.items()
    }
    (path / "deepship_paper_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (path / "deepship_paper_split_stats.json").write_text(
        json.dumps(split_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


class DeepShipMelSegmentDataset(Dataset):
    def __init__(
        self,
        segments: list[SegmentRecord],
        *,
        sample_rate: int = 16000,
        clip_duration: float = 3.0,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int = 1024,
        n_mels: int = 64,
    ) -> None:
        self.segments = segments
        self.sample_rate = sample_rate
        self.clip_samples = int(round(sample_rate * clip_duration))
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=2.0,
            center=True,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        segment = self.segments[index]
        waveform, source_sr = self._load_segment(segment)
        if source_sr != self.sample_rate:
            waveform = self._resample(waveform, source_sr)
        waveform = self._fix_length(waveform)
        mel = self.to_db(self.mel_transform(waveform))
        return mel, segment.label_index

    def _resample(self, waveform: torch.Tensor, source_sr: int) -> torch.Tensor:
        if source_sr not in self.resamplers:
            self.resamplers[source_sr] = torchaudio.transforms.Resample(
                orig_freq=source_sr,
                new_freq=self.sample_rate,
            )
        return self.resamplers[source_sr](waveform)

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        num_samples = waveform.size(-1)
        if num_samples > self.clip_samples:
            return waveform[..., : self.clip_samples]
        if num_samples < self.clip_samples:
            return torch.nn.functional.pad(waveform, (0, self.clip_samples - num_samples))
        return waveform

    @staticmethod
    def _load_segment(segment: SegmentRecord) -> tuple[torch.Tensor, int]:
        audio, sample_rate = sf.read(
            segment.path,
            start=segment.start_frame,
            frames=segment.num_frames,
            dtype="float32",
            always_2d=True,
        )
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1, keepdims=True)
        waveform = torch.from_numpy(audio[:, 0]).unsqueeze(0)
        return waveform, sample_rate
