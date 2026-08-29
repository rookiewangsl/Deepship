from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from src.data.deepship import DeepShipMelSegmentDataset, SegmentRecord
from src.data.deepship_waveform import (
    DeepShipMelWindowDataset,
    DeepShipWaveformSegmentDataset,
    RecordingBalancedEpochSampler,
    VesselBalancedEpochSampler,
    recording_representatives,
)


class DeepShipWaveformDatasetTests(unittest.TestCase):
    def _segment(
        self,
        *,
        start_frame: int,
        num_frames: int,
        sample_rate: int,
        relative_path: str = "Cargo/example.wav",
        class_name: str = "Cargo",
        label_index: int = 0,
        segment_index: int = 0,
        group_key: str = "recording-1",
        vessel_key: str = "vessel-1",
    ) -> SegmentRecord:
        return SegmentRecord(
            relative_path=relative_path,
            class_name=class_name,
            label_index=label_index,
            start_frame=start_frame,
            num_frames=num_frames,
            sample_rate=sample_rate,
            segment_index=segment_index,
            total_segments=1,
            group_key=group_key,
            vessel_key=vessel_key,
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

    def test_dynamic_crop_is_seeded_and_uses_the_whole_recording(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            audio_path = root / "Cargo" / "example.wav"
            audio_path.parent.mkdir(parents=True)
            sample_rate = 10
            waveform = np.arange(40, dtype=np.float32) / 40.0
            sf.write(audio_path, waveform, sample_rate, subtype="FLOAT")
            dataset = DeepShipWaveformSegmentDataset(
                [self._segment(start_frame=0, num_frames=5, sample_rate=sample_rate)],
                data_root=root,
                sample_rate=sample_rate,
                clip_duration=0.5,
                normalize=False,
                remove_dc=False,
                dynamic_crop=True,
            )

            first, first_mask, _ = dataset[(0, 11)]
            repeated, _, _ = dataset[(0, 11)]
            second, second_mask, _ = dataset[(0, 12)]

            np.testing.assert_allclose(first.numpy(), repeated.numpy())
            self.assertFalse(first.equal(second))
            self.assertEqual(int(first_mask.sum()), 5)
            self.assertEqual(int(second_mask.sum()), 5)
            with self.assertRaisesRegex(ValueError, "explicit deterministic crop seed"):
                dataset[0]

    def test_long_mel_window_zeroes_padding_and_returns_valid_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            audio_path = root / "Cargo" / "example.wav"
            audio_path.parent.mkdir(parents=True)
            sample_rate = 16_000
            waveform = np.sin(
                2 * np.pi * 200 * np.arange(sample_rate, dtype=np.float32) / sample_rate
            )
            sf.write(audio_path, waveform, sample_rate)
            dataset = DeepShipMelWindowDataset(
                [self._segment(start_frame=0, num_frames=sample_rate, sample_rate=sample_rate)],
                data_root=root,
                sample_rate=sample_rate,
                clip_duration=2.0,
                n_fft=64,
                win_length=64,
                hop_length=16,
                n_mels=8,
                return_index=True,
            )

            mel, label, index, valid_frames = dataset[0]

            self.assertEqual(mel.shape, (1, 8, 2001))
            self.assertEqual(valid_frames, 1001)
            self.assertEqual(label, 0)
            self.assertEqual(index, 0)
            self.assertTrue(torch.isfinite(mel).all())
            self.assertTrue(mel[..., valid_frames:].eq(0).all())

    def test_full_fixed_window_matches_existing_log_mel_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            audio_path = root / "Cargo" / "example.wav"
            audio_path.parent.mkdir(parents=True)
            sample_rate = 16_000
            waveform = np.sin(
                2 * np.pi * 320 * np.arange(sample_rate, dtype=np.float32) / sample_rate
            )
            sf.write(audio_path, waveform, sample_rate)
            rows = [
                self._segment(
                    start_frame=0,
                    num_frames=sample_rate,
                    sample_rate=sample_rate,
                )
            ]
            common = dict(
                data_root=root,
                sample_rate=sample_rate,
                clip_duration=1.0,
                n_fft=64,
                win_length=64,
                hop_length=16,
                n_mels=8,
            )
            existing = DeepShipMelSegmentDataset(rows, **common)
            windowed = DeepShipMelWindowDataset(rows, **common)

            expected, expected_label = existing[0]
            actual, actual_label, valid_frames = windowed[0]

            torch.testing.assert_close(actual, expected)
            self.assertEqual(actual_label, expected_label)
            self.assertEqual(valid_frames, expected.size(-1))

    def test_recording_representatives_are_stable_and_validate_metadata(self) -> None:
        later = self._segment(
            start_frame=30,
            num_frames=10,
            sample_rate=10,
            segment_index=2,
        )
        earlier = self._segment(
            start_frame=10,
            num_frames=10,
            sample_rate=10,
            segment_index=0,
        )
        other = self._segment(
            start_frame=0,
            num_frames=10,
            sample_rate=10,
            relative_path="Passenger/other.wav",
            class_name="Passenger",
            label_index=1,
            group_key="recording-2",
            vessel_key="vessel-2",
        )

        representatives = recording_representatives([later, other, earlier])

        self.assertEqual(
            [row.relative_path for row in representatives],
            ["Cargo/example.wav", "Passenger/other.wav"],
        )
        self.assertEqual(representatives[0].segment_index, 0)

        conflict = self._segment(
            start_frame=20,
            num_frames=10,
            sample_rate=10,
            class_name="Passenger",
            label_index=1,
            segment_index=1,
        )
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            recording_representatives([earlier, conflict])

    def test_recording_balanced_sampler_is_exact_and_reproducible(self) -> None:
        recordings = []
        for label, class_name in enumerate(("Cargo", "Passenger")):
            for recording_index in range(3):
                recordings.append(
                    self._segment(
                        start_frame=0,
                        num_frames=10,
                        sample_rate=10,
                        relative_path=f"{class_name}/{recording_index}.wav",
                        class_name=class_name,
                        label_index=label,
                        group_key=f"recording-{label}-{recording_index}",
                        vessel_key=f"vessel-{label}-{recording_index}",
                    )
                )
        sampler = RecordingBalancedEpochSampler(
            recordings,
            epoch_samples=10,
            seed=42,
        )

        epoch_one = list(sampler)
        self.assertEqual(epoch_one, list(sampler))
        class_counts = {
            label: sum(recordings[index].label_index == label for index, _ in epoch_one)
            for label in (0, 1)
        }
        self.assertEqual(class_counts, {0: 5, 1: 5})
        for label in (0, 1):
            draws = [
                sum(index == recording_index for index, _ in epoch_one)
                for recording_index, row in enumerate(recordings)
                if row.label_index == label
            ]
            self.assertLessEqual(max(draws) - min(draws), 1)

        report = sampler.exposure_report()
        self.assertEqual(report["classes"], {"Cargo": 5, "Passenger": 5})
        self.assertEqual(report["unique_recordings"], 6)
        self.assertAlmostEqual(float(report["recording_repeat_rate"]), 0.4)
        for summary in report["recording_draws_by_class"].values():
            self.assertLessEqual(int(summary["max"]) - int(summary["min"]), 1)

        sampler.set_epoch(2)
        self.assertNotEqual(epoch_one, list(sampler))

    def test_vessel_balanced_sampler_balances_vessels_then_recordings(self) -> None:
        specifications = [
            (0, "Cargo", "cargo-a", 3),
            (0, "Cargo", "cargo-b", 1),
            (1, "Passenger", "passenger-a", 2),
            (1, "Passenger", "passenger-b", 1),
        ]
        recordings = []
        for label, class_name, vessel_key, recording_count in specifications:
            for recording_index in range(recording_count):
                recordings.append(
                    self._segment(
                        start_frame=0,
                        num_frames=10,
                        sample_rate=10,
                        relative_path=(
                            f"{class_name}/{vessel_key}-{recording_index}.wav"
                        ),
                        class_name=class_name,
                        label_index=label,
                        group_key=vessel_key,
                        vessel_key=vessel_key,
                    )
                )
        sampler = VesselBalancedEpochSampler(
            recordings,
            epoch_samples=20,
            seed=42,
        )

        requests = list(sampler)
        vessel_counts = {
            vessel_key: sum(
                recordings[index].vessel_key == vessel_key for index, _ in requests
            )
            for _, _, vessel_key, _ in specifications
        }
        self.assertEqual(set(vessel_counts.values()), {5})
        for _, _, vessel_key, _ in specifications:
            per_recording = [
                sum(index == recording_index for index, _ in requests)
                for recording_index, row in enumerate(recordings)
                if row.vessel_key == vessel_key
            ]
            self.assertLessEqual(max(per_recording) - min(per_recording), 1)

        report = sampler.exposure_report()
        for summary in report["vessel_draws_by_class"].values():
            self.assertEqual(summary["min"], summary["max"])


if __name__ == "__main__":
    unittest.main()
