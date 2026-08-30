from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import soundfile as sf

from src.data.belgian_ais import BelgianRecord, canonical_sha256
from src.pipelines.mel_ml.train_belgian_macnna_global import (
    BelgianTrainConfig,
    train,
)


class BelgianTrainingTests(unittest.TestCase):
    def test_validation_only_smoke_writes_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "audio"
            data_root.mkdir()
            rows = []
            classes = (("Cargo", 0, "Cargo"), ("Passenger", 1, "Passenger"), ("Tank", 2, "Tanker"), ("Tug", 3, "Tug"))
            for split, date, offset in (("train", "2022-01-01", 0), ("val", "2022-01-02", 100)):
                for class_name, label, vessel_type in classes:
                    relative_path = f"{split}-{class_name}.wav"
                    waveform = np.full(1600, (label + 1) * 0.01, dtype=np.float32)
                    sf.write(data_root / relative_path, waveform, 16_000)
                    record = BelgianRecord(
                        relative_path=relative_path,
                        class_name=class_name,
                        label_index=label,
                        vessel_type=vessel_type,
                        official_split="train" if split == "train" else "val",
                        event_time=f"{date} 00:00:00+00:00",
                        calendar_date=date,
                        station="Grafton",
                        distance_km=1.0,
                        activity="underway-using-engine",
                    )
                    rows.append({**asdict(record), "split": split})
            manifest = {
                "schema_version": 1,
                "experiment_id": "belgian_attention_v1",
                "protocol": "utc_date_disjoint",
                "fold": 1,
                "fold_seed": 42,
                "source_metadata_sha256": "synthetic",
                "filters": {"max_distance_km": 5.0},
                "test_policy": "sealed",
                "records": rows,
            }
            manifest["manifest_sha256"] = canonical_sha256(manifest)
            manifest_path = root / "split_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output_root = root / "run"
            result = train(
                BelgianTrainConfig(
                    data_root=str(data_root),
                    output_root=str(output_root),
                    split_manifest=str(manifest_path),
                    model_variant="g0",
                    seed=42,
                    clip_duration=0.1,
                    source_sample_rate=16_000,
                    n_fft=64,
                    win_length=64,
                    hop_length=32,
                    batch_size=4,
                    eval_batch_size=4,
                    gradient_accumulation_steps=1,
                    epochs=1,
                    warmup_epochs=0,
                    precision="fp32",
                    num_workers=0,
                    max_train_batches=1,
                    max_eval_batches=1,
                    allow_experiment_overrides=True,
                    device="cpu",
                )
            )
            self.assertEqual(result["status"], "validation_complete")
            self.assertFalse(result["test_evaluated"])
            self.assertTrue((output_root / "models" / "belgian_best.pt").is_file())
            self.assertTrue((output_root / "models" / "belgian_last.pt").is_file())
            self.assertTrue(
                (output_root / "predictions" / "validation_best_predictions.csv").is_file()
            )
            self.assertTrue(
                (output_root / "metrics" / "validation_best_date_balanced_metrics.json").is_file()
            )
            completion = json.loads(
                (output_root / "reports" / "run_complete.json").read_text(encoding="utf-8")
            )
            self.assertFalse(completion["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
