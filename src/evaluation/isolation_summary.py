from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Mapping

from src.pipelines.mel_ml.isolation_experiment import find_training_config_mismatches


METRIC_LEVELS = ("segment", "recording", "vessel")
METRIC_NAMES = ("accuracy", "macro_f1", "weighted_f1")


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _expected_manifest_hashes(protocol_root: Path, protocols: Iterable[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for protocol in protocols:
        manifest = _load_json(protocol_root / protocol / "split_manifest.json")
        manifest_protocol = manifest.get("protocol")
        if manifest_protocol != protocol:
            raise ValueError(
                f"Manifest protocol mismatch for {protocol}: found {manifest_protocol!r}"
            )
        manifest_hash = manifest.get("manifest_sha256")
        if not isinstance(manifest_hash, str) or not manifest_hash:
            raise ValueError(f"Manifest SHA-256 is missing for {protocol}")
        hashes[protocol] = manifest_hash
    return hashes


def _validate_confusion_matrix(value: object, *, source: Path) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Invalid confusion matrix in {source}")
    size = len(value)
    matrix: list[list[int]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(f"Confusion matrix is not square in {source}")
        converted = []
        for item in row:
            if not isinstance(item, int) or item < 0:
                raise ValueError(f"Confusion matrix contains an invalid count in {source}")
            converted.append(item)
        matrix.append(converted)
    return matrix


def _read_run(
    run_root: Path,
    *,
    protocol: str,
    seed: int,
    experiment: Mapping[str, object],
    expected_manifest_hash: str,
) -> dict[str, object]:
    reports = run_root / "reports"
    complete = _load_json(reports / "run_complete.json")
    run_config = _load_json(reports / "deepship_macnna_run_config.json")
    environment = _load_json(reports / "environment.json")

    if complete.get("status") != "complete":
        raise ValueError(f"Run is not complete: {run_root}")
    for source_name, source in (("run_complete", complete), ("run_config", run_config)):
        if source.get("protocol") != protocol or source.get("seed") != seed:
            raise ValueError(
                f"{source_name} identity mismatch in {run_root}: expected {protocol}/seed{seed}"
            )
        if source.get("split_manifest_sha256") != expected_manifest_hash:
            raise ValueError(f"Unexpected split manifest hash in {run_root}/{source_name}")

    mismatches = find_training_config_mismatches(run_config, experiment)
    if mismatches:
        raise ValueError(
            f"Run differs from the frozen experiment: {run_root}\n- " + "\n- ".join(mismatches)
        )
    if run_config.get("experiment_config_mismatches") != []:
        raise ValueError(f"Run records experiment overrides and is not formal: {run_root}")
    if run_config.get("allow_experiment_overrides") is not False:
        raise ValueError(f"Formal run enabled experiment overrides: {run_root}")

    model = experiment["model"]
    assert isinstance(model, Mapping)
    if run_config.get("num_parameters") != model["expected_num_parameters"]:
        raise ValueError(f"Model parameter count mismatch in {run_root}")

    git_commit = environment.get("git_commit")
    if not isinstance(git_commit, str) or not git_commit or git_commit == "unknown":
        raise ValueError(f"Git commit is unavailable in {run_root}")
    if environment.get("git_worktree_dirty") is not False:
        raise ValueError(f"Formal run was not produced from a clean git worktree: {run_root}")

    row: dict[str, object] = {
        "run_root": str(run_root),
        "protocol": protocol,
        "seed": seed,
        "git_commit": git_commit,
        "split_manifest_sha256": expected_manifest_hash,
        "num_parameters": run_config["num_parameters"],
        "best_epoch": complete.get("best_epoch"),
        "best_val_acc": complete.get("best_val_acc"),
    }
    matrices: dict[str, list[list[int]]] = {}
    for level in METRIC_LEVELS:
        metrics_path = run_root / "metrics" / f"{level}_metrics.json"
        metrics = _load_json(metrics_path)
        for metric_name in METRIC_NAMES:
            value = metrics.get(metric_name)
            if not isinstance(value, (int, float)):
                raise ValueError(f"Missing {level} {metric_name} in {metrics_path}")
            row[f"{level}_{metric_name}"] = float(value)
        matrices[level] = _validate_confusion_matrix(
            metrics.get("confusion_matrix"),
            source=metrics_path,
        )
    row["confusion_matrices"] = matrices
    return row


def _sum_matrices(matrices: list[list[list[int]]]) -> list[list[int]]:
    if not matrices:
        raise ValueError("Cannot aggregate an empty confusion-matrix list")
    size = len(matrices[0])
    total = [[0 for _ in range(size)] for _ in range(size)]
    for matrix in matrices:
        if len(matrix) != size or any(len(row) != size for row in matrix):
            raise ValueError("Confusion matrix dimensions differ between runs")
        for row_index in range(size):
            for column_index in range(size):
                total[row_index][column_index] += matrix[row_index][column_index]
    return total


def summarize_isolation_runs(
    runs_root: str | Path,
    experiment_config: str | Path,
    protocol_root: str | Path,
) -> dict[str, object]:
    runs_path = Path(runs_root).expanduser().resolve()
    experiment = _load_json(Path(experiment_config).expanduser().resolve())
    split = experiment["split"]
    training = experiment["training"]
    assert isinstance(split, Mapping)
    assert isinstance(training, Mapping)
    protocols = [str(value) for value in split["protocols"]]
    seeds = [int(value) for value in training["model_seeds"]]
    manifest_hashes = _expected_manifest_hashes(
        Path(protocol_root).expanduser().resolve(),
        protocols,
    )

    rows = []
    for protocol in protocols:
        for seed in seeds:
            run_root = runs_path / f"{protocol}_seed{seed}"
            rows.append(
                _read_run(
                    run_root,
                    protocol=protocol,
                    seed=seed,
                    experiment=experiment,
                    expected_manifest_hash=manifest_hashes[protocol],
                )
            )

    commits = {str(row["git_commit"]) for row in rows}
    if len(commits) != 1:
        raise ValueError(f"Formal runs were produced by different git commits: {sorted(commits)}")

    summaries = []
    aggregate_confusion_matrices: dict[str, dict[str, list[list[int]]]] = {}
    for protocol in protocols:
        protocol_rows = [row for row in rows if row["protocol"] == protocol]
        aggregate_confusion_matrices[protocol] = {}
        for level in METRIC_LEVELS:
            matrices = [
                row["confusion_matrices"][level]  # type: ignore[index]
                for row in protocol_rows
            ]
            aggregate_confusion_matrices[protocol][level] = _sum_matrices(matrices)
            for metric_name in METRIC_NAMES:
                values = [float(row[f"{level}_{metric_name}"]) for row in protocol_rows]
                summaries.append(
                    {
                        "protocol": protocol,
                        "level": level,
                        "metric": metric_name,
                        "n": len(values),
                        "mean": mean(values),
                        "std": stdev(values) if len(values) > 1 else 0.0,
                        "min": min(values),
                        "max": max(values),
                    }
                )

    serializable_rows = []
    for row in rows:
        serializable_rows.append({key: value for key, value in row.items() if key != "confusion_matrices"})
    return {
        "status": "complete",
        "experiment_id": experiment.get("experiment_id"),
        "git_commit": next(iter(commits)),
        "protocols": protocols,
        "seeds": seeds,
        "run_count": len(rows),
        "standard_deviation": "sample (n-1)",
        "runs": serializable_rows,
        "summary": summaries,
        "aggregate_confusion_matrices": aggregate_confusion_matrices,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_isolation_summary(summary: Mapping[str, object], output_dir: str | Path) -> None:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "isolation_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_path / "individual_runs.csv", list(summary["runs"]))  # type: ignore[arg-type]
    _write_csv(output_path / "protocol_summary.csv", list(summary["summary"]))  # type: ignore[arg-type]

    lookup = defaultdict(dict)
    for row in summary["summary"]:  # type: ignore[union-attr]
        lookup[(row["protocol"], row["level"])][row["metric"]] = row
    lines = [
        "# DeepShip isolation comparison",
        "",
        "Mean ± sample standard deviation across model seeds 42, 43, and 44.",
        "",
        "| Protocol | Level | Accuracy | Macro-F1 |",
        "|---|---|---:|---:|",
    ]
    for protocol in summary["protocols"]:  # type: ignore[union-attr]
        for level in METRIC_LEVELS:
            accuracy = lookup[(protocol, level)]["accuracy"]
            macro_f1 = lookup[(protocol, level)]["macro_f1"]
            lines.append(
                f"| {protocol} | {level} | "
                f"{accuracy['mean']:.4f} ± {accuracy['std']:.4f} | "
                f"{macro_f1['mean']:.4f} ± {macro_f1['std']:.4f} |"
            )
    (output_path / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
