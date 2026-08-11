from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation.isolation_summary import summarize_isolation_runs, write_isolation_summary


PROTOCOLS = ["segment_level", "recording_disjoint", "vessel_name_disjoint"]
SEEDS = [42, 43, 44]


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def experiment_payload() -> dict[str, object]:
    return {
        "experiment_id": "test_v1",
        "model": {"expected_num_parameters": 532166},
        "features": {
            "clip_duration_seconds": 3.0,
            "n_fft": 1024,
            "hop_length": 512,
            "win_length": 1024,
            "n_mels": 64,
            "highpass_freq": None,
        },
        "split": {
            "protocols": PROTOCOLS,
            "target_segments_per_class": {"train": 3500, "val": 1000, "test": 500},
        },
        "training": {
            "model_seeds": SEEDS,
            "batch_size": 16,
            "epochs": 100,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "min_learning_rate": 0.00001,
            "warmup_epochs": 10,
            "early_stopping_patience": 10,
        },
    }


def run_config(protocol: str, seed: int, manifest_hash: str) -> dict[str, object]:
    return {
        "protocol": protocol,
        "seed": seed,
        "split_manifest_sha256": manifest_hash,
        "num_parameters": 532166,
        "experiment_config_mismatches": [],
        "allow_experiment_overrides": False,
        "clip_duration": 3.0,
        "n_fft": 1024,
        "hop_length": 512,
        "win_length": 1024,
        "n_mels": 64,
        "highpass_freq": None,
        "batch_size": 16,
        "epochs": 100,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "min_learning_rate": 0.00001,
        "warmup_epochs": 10,
        "early_stopping_patience": 10,
        "train_per_class": 3500,
        "val_per_class": 1000,
        "test_per_class": 500,
        "max_train_batches": None,
        "max_eval_batches": None,
    }


class IsolationSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runs = self.root / "runs"
        self.protocols = self.root / "protocols"
        self.config = self.root / "experiment.json"
        write_json(self.config, experiment_payload())
        for protocol in PROTOCOLS:
            manifest_hash = f"hash-{protocol}"
            write_json(
                self.protocols / protocol / "split_manifest.json",
                {"protocol": protocol, "manifest_sha256": manifest_hash},
            )
            for offset, seed in enumerate(SEEDS):
                run = self.runs / f"{protocol}_seed{seed}"
                complete = {
                    "status": "complete",
                    "protocol": protocol,
                    "seed": seed,
                    "best_epoch": 10 + offset,
                    "best_val_acc": 0.8 + offset * 0.01,
                    "split_manifest_sha256": manifest_hash,
                }
                write_json(run / "reports" / "run_complete.json", complete)
                write_json(
                    run / "reports" / "deepship_macnna_run_config.json",
                    run_config(protocol, seed, manifest_hash),
                )
                write_json(
                    run / "reports" / "environment.json",
                    {"git_commit": "abc123", "git_worktree_dirty": False},
                )
                for level_index, level in enumerate(("segment", "recording", "vessel")):
                    score = 0.7 + offset * 0.01 + level_index * 0.02
                    write_json(
                        run / "metrics" / f"{level}_metrics.json",
                        {
                            "accuracy": score,
                            "macro_f1": score - 0.01,
                            "weighted_f1": score - 0.005,
                            "confusion_matrix": [[2, 0], [0, 3]],
                        },
                    )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_summarizes_exact_nine_runs_and_writes_outputs(self) -> None:
        summary = summarize_isolation_runs(self.runs, self.config, self.protocols)

        self.assertEqual(summary["run_count"], 9)
        self.assertEqual(summary["git_commit"], "abc123")
        self.assertEqual(len(summary["summary"]), 27)
        matrix = summary["aggregate_confusion_matrices"]["segment_level"]["segment"]
        self.assertEqual(matrix, [[6, 0], [0, 9]])

        output = self.root / "summary"
        write_isolation_summary(summary, output)
        self.assertTrue((output / "isolation_comparison_summary.json").is_file())
        self.assertTrue((output / "individual_runs.csv").is_file())
        self.assertTrue((output / "protocol_summary.csv").is_file())
        self.assertIn("segment_level", (output / "comparison_table.md").read_text())

    def test_rejects_run_from_different_commit(self) -> None:
        write_json(
            self.runs / "recording_disjoint_seed43" / "reports" / "environment.json",
            {"git_commit": "different", "git_worktree_dirty": False},
        )

        with self.assertRaisesRegex(ValueError, "different git commits"):
            summarize_isolation_runs(self.runs, self.config, self.protocols)

    def test_rejects_smoke_override(self) -> None:
        path = (
            self.runs
            / "segment_level_seed42"
            / "reports"
            / "deepship_macnna_run_config.json"
        )
        config = json.loads(path.read_text())
        config["epochs"] = 1
        write_json(path, config)

        with self.assertRaisesRegex(ValueError, "differs from the frozen experiment"):
            summarize_isolation_runs(self.runs, self.config, self.protocols)

    def test_rejects_dirty_worktree_run(self) -> None:
        path = self.runs / "vessel_name_disjoint_seed44" / "reports" / "environment.json"
        write_json(path, {"git_commit": "abc123", "git_worktree_dirty": True})

        with self.assertRaisesRegex(ValueError, "clean git worktree"):
            summarize_isolation_runs(self.runs, self.config, self.protocols)


if __name__ == "__main__":
    unittest.main()
