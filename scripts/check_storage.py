"""Validate the external DeepShip storage layout without modifying it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.pathing import default_deepship_root, resolve_path  # noqa: E402


CLASS_NAMES = ("Cargo", "Passenger", "Tank", "Tug")


def infer_storage_root(data_root: Path) -> Path:
    resolved = data_root.resolve()
    if resolved.parent.name == "datasets":
        return resolved.parent.parent
    return resolved.parent


def directory_summary(path: Path) -> dict[str, int | bool]:
    files = (
        [
            item
            for item in path.rglob("*")
            if item.is_file() and not item.name.startswith("._")
        ]
        if path.exists()
        else []
    )
    return {
        "exists": path.is_dir(),
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=default_deepship_root())
    parser.add_argument("--storage-root")
    arguments = parser.parse_args()

    data_root = resolve_path(arguments.data_root)
    storage_root = (
        resolve_path(arguments.storage_root)
        if arguments.storage_root is not None
        else infer_storage_root(data_root)
    )
    errors: list[str] = []
    class_counts: dict[str, int] = {}
    audio_bytes = 0
    representative_audio: dict[str, object] | None = None

    if not data_root.is_dir():
        errors.append(f"dataset root is unavailable: {data_root}")
    else:
        for class_name in CLASS_NAMES:
            class_directory = data_root / class_name
            files = (
                [
                    path
                    for path in sorted(class_directory.rglob("*.wav"))
                    if not path.name.startswith("._")
                ]
                if class_directory.is_dir()
                else []
            )
            class_counts[class_name] = len(files)
            audio_bytes += sum(path.stat().st_size for path in files)
            if not files:
                errors.append(f"no WAV files found for class {class_name}")
            elif representative_audio is None:
                info = sf.info(str(files[0]))
                representative_audio = {
                    "path": str(files[0]),
                    "sample_rate": info.samplerate,
                    "frames": info.frames,
                    "channels": info.channels,
                }

    if not storage_root.is_dir():
        errors.append(f"storage root is unavailable: {storage_root}")
        disk = None
    else:
        usage = shutil.disk_usage(storage_root)
        disk = {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }

    runs_root = storage_root / "runs"
    if not runs_root.is_dir() or not os.access(runs_root, os.W_OK):
        errors.append(f"runs directory is missing or not writable: {runs_root}")

    report = {
        "status": "passed" if not errors else "failed",
        "data_root": str(data_root),
        "storage_root": str(storage_root),
        "class_counts": class_counts,
        "recordings": sum(class_counts.values()),
        "audio_bytes": audio_bytes,
        "representative_audio": representative_audio,
        "shipsear": directory_summary(storage_root / "datasets" / "ShipsEar"),
        "precomputed": directory_summary(storage_root / "precomputed"),
        "runs_writable": runs_root.is_dir() and os.access(runs_root, os.W_OK),
        "disk": disk,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
