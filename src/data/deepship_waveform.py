from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
from statistics import median
from typing import Iterator

import soundfile as sf
import torch
import torch.nn.functional as nnf
import torchaudio
from torch.utils.data import Dataset, Sampler

from src.data.deepship import SegmentRecord
from src.utils.pathing import resolve_manifest_path, resolve_path


class DeepShipWaveformSegmentDataset(Dataset):
    """Load waveform windows anchored by a frozen DeepShip segment manifest.

    The frozen manifests contain three-second anchors. ``clip_duration`` may be
    longer: the dataset centers the requested window on the anchor while keeping
    the recording and split assignment unchanged. This makes 3/10/20/30-second
    context ablations possible without regenerating the established isolation
    protocol.
    """

    def __init__(
        self,
        segments: list[SegmentRecord],
        *,
        data_root: str | Path,
        sample_rate: int = 16000,
        clip_duration: float = 20.0,
        normalize: bool = True,
        remove_dc: bool = True,
        return_index: bool = False,
        dynamic_crop: bool = False,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if clip_duration <= 0:
            raise ValueError("clip_duration must be positive")

        self.segments = segments
        self.data_root = resolve_path(data_root)
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.clip_samples = int(round(sample_rate * clip_duration))
        self.normalize = normalize
        self.remove_dc = remove_dc
        self.return_index = return_index
        self.dynamic_crop = dynamic_crop
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}
        self.audio_info: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(
        self,
        index: int | tuple[int, int],
    ) -> (
        tuple[torch.Tensor, torch.Tensor, int]
        | tuple[torch.Tensor, torch.Tensor, int, int]
    ):
        crop_seed = None
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError("Dynamic sample request must contain index and crop seed")
            segment_index, crop_seed = int(index[0]), int(index[1])
        else:
            segment_index = int(index)
        if self.dynamic_crop and crop_seed is None:
            raise ValueError("Dynamic cropping requires an explicit deterministic crop seed")
        if not self.dynamic_crop and crop_seed is not None:
            raise ValueError("Fixed-anchor datasets do not accept dynamic crop requests")

        segment = self.segments[segment_index]
        waveform, source_sample_rate = self._load_window(segment, crop_seed=crop_seed)
        if source_sample_rate != self.sample_rate:
            waveform = self._resample(waveform, source_sample_rate)

        valid_samples = min(waveform.size(-1), self.clip_samples)
        waveform = self._fix_length(waveform)
        attention_mask = torch.zeros(self.clip_samples, dtype=torch.long)
        attention_mask[:valid_samples] = 1

        if valid_samples > 0:
            valid_waveform = waveform[..., :valid_samples]
            if self.remove_dc:
                waveform[..., :valid_samples] = valid_waveform - valid_waveform.mean(
                    dim=-1,
                    keepdim=True,
                )
                valid_waveform = waveform[..., :valid_samples]
            if self.normalize:
                scale = valid_waveform.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-7)
                waveform[..., :valid_samples] = valid_waveform / scale
        waveform[..., valid_samples:] = 0.0

        values = waveform.squeeze(0)
        if self.return_index:
            return values, attention_mask, segment.label_index, segment_index
        return values, attention_mask, segment.label_index

    def _resample(self, waveform: torch.Tensor, source_sample_rate: int) -> torch.Tensor:
        if source_sample_rate not in self.resamplers:
            self.resamplers[source_sample_rate] = torchaudio.transforms.Resample(
                orig_freq=source_sample_rate,
                new_freq=self.sample_rate,
            )
        return self.resamplers[source_sample_rate](waveform)

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        num_samples = waveform.size(-1)
        if num_samples > self.clip_samples:
            return waveform[..., : self.clip_samples]
        if num_samples < self.clip_samples:
            return nnf.pad(waveform, (0, self.clip_samples - num_samples))
        return waveform

    def _load_window(
        self,
        segment: SegmentRecord,
        *,
        crop_seed: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        audio_path = resolve_manifest_path(self.data_root, segment.relative_path)
        if segment.relative_path not in self.audio_info:
            self.audio_info[segment.relative_path] = sf.info(str(audio_path))
        info = self.audio_info[segment.relative_path]

        requested_frames = int(round(info.samplerate * self.clip_duration))
        max_start = max(0, info.frames - requested_frames)
        if self.dynamic_crop:
            if crop_seed is None:
                raise ValueError("Dynamic cropping requires a crop seed")
            start_frame = random.Random(crop_seed).randrange(max_start + 1)
        else:
            anchor_center = segment.start_frame + segment.num_frames // 2
            start_frame = anchor_center - requested_frames // 2
            start_frame = min(max(0, start_frame), max_start)
        frames_to_read = min(requested_frames, info.frames - start_frame)

        audio, source_sample_rate = sf.read(
            audio_path,
            start=start_frame,
            frames=frames_to_read,
            dtype="float32",
            always_2d=True,
        )
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1, keepdims=True)
        waveform = torch.from_numpy(audio[:, 0]).unsqueeze(0)
        return waveform, source_sample_rate


def recording_representatives(segments: list[SegmentRecord]) -> list[SegmentRecord]:
    """Return one validated, stable representative for every recording."""

    grouped: dict[str, list[SegmentRecord]] = defaultdict(list)
    for segment in segments:
        grouped[segment.relative_path].append(segment)
    representatives = []
    for relative_path, rows in sorted(grouped.items()):
        labels = {(row.class_name, row.label_index) for row in rows}
        vessels = {row.vessel_key for row in rows}
        groups = {row.group_key for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Recording has conflicting labels: {relative_path}")
        if len(vessels) != 1:
            raise ValueError(f"Recording has conflicting vessel keys: {relative_path}")
        if len(groups) != 1:
            raise ValueError(f"Recording has conflicting group keys: {relative_path}")
        representatives.append(
            min(rows, key=lambda row: (row.segment_index, row.start_frame))
        )
    if not representatives:
        raise ValueError("At least one recording is required")
    return representatives


class RecordingBalancedEpochSampler(Sampler[tuple[int, int]]):
    """Build a deterministic class→recording-balanced dynamic-crop epoch.

    Every class receives the same number of draws up to an unavoidable one-sample
    remainder. Within a class, shuffled cycles make recording exposure differ by
    at most one. Each request carries its own crop seed, so DataLoader worker count
    and persistent-worker state do not change crop locations.
    """

    def __init__(
        self,
        recordings: list[SegmentRecord],
        *,
        epoch_samples: int,
        seed: int,
    ) -> None:
        if epoch_samples <= 0:
            raise ValueError("epoch_samples must be positive")
        if not recordings:
            raise ValueError("At least one recording is required")
        paths = [row.relative_path for row in recordings]
        if len(paths) != len(set(paths)):
            raise ValueError("RecordingBalancedEpochSampler requires unique recordings")
        self.recordings = recordings
        self.epoch_samples = epoch_samples
        self.seed = seed
        self.epoch = 1
        self.indexes_by_class: dict[int, list[int]] = defaultdict(list)
        for index, recording in enumerate(recordings):
            self.indexes_by_class[recording.label_index].append(index)
        if epoch_samples < len(self.indexes_by_class):
            raise ValueError("epoch_samples cannot be smaller than the number of classes")

    def set_epoch(self, epoch: int) -> None:
        if epoch <= 0:
            raise ValueError("epoch must be positive")
        self.epoch = epoch

    def _requests(self) -> list[tuple[int, int]]:
        labels = sorted(self.indexes_by_class)
        base_count, remainder = divmod(self.epoch_samples, len(labels))
        requests: list[tuple[int, int]] = []
        for label_position, label in enumerate(labels):
            target = base_count + (1 if label_position < remainder else 0)
            indexes = self.indexes_by_class[label]
            rng = random.Random(
                (self.seed + 1) * 1_000_003 + self.epoch * 10_007 + label * 101
            )
            selected: list[int] = []
            while len(selected) < target:
                cycle = indexes[:]
                rng.shuffle(cycle)
                selected.extend(cycle[: target - len(selected)])
            requests.extend((index, rng.getrandbits(63)) for index in selected)
        random.Random((self.seed + 1) * 97_409 + self.epoch * 65_537).shuffle(
            requests
        )
        return requests

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(self._requests())

    def __len__(self) -> int:
        return self.epoch_samples

    @staticmethod
    def _count_summary(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"min": 0, "median": 0.0, "mean": 0.0, "max": 0}
        return {
            "min": min(values),
            "median": float(median(values)),
            "mean": sum(values) / len(values),
            "max": max(values),
        }

    def exposure_report(self) -> dict[str, object]:
        requests = self._requests()
        recording_counts = Counter(index for index, _ in requests)
        class_counts = Counter(
            self.recordings[index].class_name for index, _ in requests
        )
        vessel_counts = Counter(
            self.recordings[index].vessel_key for index, _ in requests
        )
        return {
            "epoch": self.epoch,
            "seed": self.seed,
            "samples": len(requests),
            "classes": dict(sorted(class_counts.items())),
            "unique_recordings": len(recording_counts),
            "unique_vessels": len(vessel_counts),
            "recording_repeat_rate": 1.0 - len(recording_counts) / len(requests),
            "vessel_repeat_rate": 1.0 - len(vessel_counts) / len(requests),
            "recording_draws": self._count_summary(list(recording_counts.values())),
            "vessel_draws": self._count_summary(list(vessel_counts.values())),
        }
