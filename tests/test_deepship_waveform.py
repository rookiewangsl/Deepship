from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from src.data.deepship import SegmentRecord
from src.data.deepship_waveform import DeepShipWaveformSegmentDataset


class DeepShipWaveformDatasetTests(unittest.TestCase):
    def _segment(self, *, start_frame: int, num_frames: int, sample_rate: int) -> SegmentRecord:
        return SegmentRecord(
            relative_path="Cargo/example.wav",
            class_name="Cargo",
            label_index=0,
            start_frame=start_frame,
            num_frames=num_frames,
            sample_rate=sample_rate,
            segment_index=0,
            total_segments=1,
            group_key="recording-1",
            vessel_key="vessel-1",
        )

    def test_long_context_is_centered_on_manifest_anchor_and_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            audio_path = root / "Cargo" / "example.wav"
            audio_path.parent.mkdir(parents=True)
            sample_rate = 8000
            time = np.arange(sample_rate * 5, dtype=np.float32) / sample_rate
            waveform = 0.5 + np.sin(2 * np.pi * 100 * time)
            sf.write(audio_path, waveform, sample_rate)

            dataset = DeepShipWaveformSegmentDataset(
                [
                    self._segment(
                        start_frame=sample_rate * 2,
                        num_frames=sample_rate,
                        sample_rate=sample_rate,
                    )
                ],
                data_root=root,
                sample_rate=16000,
                clip_duration=4.0,
            )
            values, attention_mask, label = dataset[0]

            self.assertEqual(values.shape, (64000,))
            self.assertEqual(attention_mask.shape, (64000,))
            self.assertEqual(int(attention_mask.sum()), 64000)
            self.assertEqual(label, 0)
            self.assertAlmostEqual(float(values.mean()), 0.0, places=4)
            self.assertAlmostEqual(float(values.std(unbiased=False)), 1.0, places=3)

    def test_short_recording_is_zero_padded_and_masked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            audio_path = root / "Cargo" / "example.wav"
            audio_path.parent.mkdir(parents=True)
            sample_rate = 16000
            sf.write(audio_path, np.ones(sample_rate, dtype=np.float32), sample_rate)

            dataset = DeepShipWaveformSegmentDataset(
                [self._segment(start_frame=0, num_frames=sample_rate, sample_rate=sample_rate)],
                data_root=root,
                sample_rate=sample_rate,
                clip_duration=3.0,
            )
            values, attention_mask, _ = dataset[0]

            self.assertEqual(int(attention_mask.sum()), sample_rate)
            self.assertTrue(values[sample_rate:].eq(0).all())


if __name__ == "__main__":
    unittest.main()
