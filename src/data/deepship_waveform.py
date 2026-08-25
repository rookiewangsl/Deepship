from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as nnf
import torchaudio
from torch.utils.data import Dataset

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
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}
        self.audio_info: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(
        self,
        index: int,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, int]
        | tuple[torch.Tensor, torch.Tensor, int, int]
    ):
        segment = self.segments[index]
        waveform, source_sample_rate = self._load_window(segment)
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
            return values, attention_mask, segment.label_index, index
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

    def _load_window(self, segment: SegmentRecord) -> tuple[torch.Tensor, int]:
        audio_path = resolve_manifest_path(self.data_root, segment.relative_path)
        if segment.relative_path not in self.audio_info:
            self.audio_info[segment.relative_path] = sf.info(str(audio_path))
        info = self.audio_info[segment.relative_path]

        requested_frames = int(round(info.samplerate * self.clip_duration))
        anchor_center = segment.start_frame + segment.num_frames // 2
        start_frame = anchor_center - requested_frames // 2
        max_start = max(0, info.frames - requested_frames)
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
