from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT
    / "protocols"
    / "isolation_comparison_v1"
    / "vessel_name_disjoint"
    / "split_manifest.json"
)
GIB = 1024**3


def resolve_from_project(raw_path: str | os.PathLike[str]) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def nearest_existing_parent(path: Path) -> Path | None:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() else None


def package_report() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    packages: dict[str, dict[str, Any]] = {}
    imported: dict[str, Any] = {}
    for import_name, display_name in (
        ("matplotlib", "matplotlib"),
        ("numpy", "numpy"),
        ("soundfile", "soundfile"),
        ("sklearn", "scikit-learn"),
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("transformers", "transformers"),
    ):
        try:
            module = importlib.import_module(import_name)
            imported[import_name] = module
            packages[display_name] = {
                "available": True,
                "version": str(getattr(module, "__version__", "unknown")),
            }
        except Exception as error:  # noqa: BLE001 - preflight must report binary import errors
            packages[display_name] = {
                "available": False,
                "error": f"{type(error).__name__}: {error}",
            }
    return packages, imported


def major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)", version)
    return (int(match.group(1)), int(match.group(2))) if match else None


def cuda_report(torch: Any) -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    report: dict[str, Any] = {
        "available": available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": int(torch.cuda.device_count()),
    }
    if not available:
        return report

    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_gib": round(properties.total_memory / GIB, 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    report["devices"] = devices
    report["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    return report


def manifest_report(manifest_path: Path, data_root: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        return {"available": False, "path": str(manifest_path)}

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "available": False,
            "path": str(manifest_path),
            "error": f"{type(error).__name__}: {error}",
        }

    segments = manifest.get("segments")
    if not isinstance(segments, list):
        return {
            "available": False,
            "path": str(manifest_path),
            "error": "manifest.segments is not a list",
        }

    split_counts: Counter[str] = Counter()
    relative_paths: set[str] = set()
    invalid_paths: list[str] = []
    missing_files: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            invalid_paths.append("<non-object segment>")
            continue
        split_counts[str(segment.get("split", ""))] += 1
        relative_path = str(segment.get("relative_path", ""))
        relative_paths.add(relative_path)
    for relative_path in sorted(relative_paths):
        candidate = (data_root / relative_path).resolve()
        try:
            candidate.relative_to(data_root)
        except ValueError:
            invalid_paths.append(relative_path)
            continue
        if not candidate.is_file():
            missing_files.append(relative_path)

    return {
        "available": True,
        "path": str(manifest_path),
        "protocol": manifest.get("protocol"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "segments": len(segments),
        "segments_by_split": dict(sorted(split_counts.items())),
        "referenced_recordings": len(relative_paths),
        "invalid_paths": len(invalid_paths),
        "invalid_path_examples": invalid_paths[:5],
        "missing_recordings": len(missing_files),
        "missing_recording_examples": missing_files[:5],
    }


def storage_report(path: Path) -> dict[str, Any]:
    parent = nearest_existing_parent(path)
    if parent is None:
        return {"path": str(path), "existing_parent": None, "writable": False}
    usage = shutil.disk_usage(parent)
    return {
        "path": str(path),
        "existing_parent": str(parent),
        "writable": os.access(parent, os.W_OK),
        "free_gib": round(usage.free / GIB, 2),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check the Linux/CUDA/data prerequisites for the DeepShip Conformer baseline "
            "without downloading a model or starting training."
        )
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DEEPSHIP_DATA_ROOT", str(ROOT / "DeepShip")),
    )
    parser.add_argument("--split-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional intended run directory; its nearest existing parent is checked.",
    )
    parser.add_argument("--min-cache-free-gib", type=float, default=5.0)
    parser.add_argument("--min-output-free-gib", type=float, default=20.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = resolve_from_project(args.data_root)
    manifest_path = resolve_from_project(args.split_manifest)
    packages, imported = package_report()

    report: dict[str, Any] = {
        "status": "pending",
        "project_root": str(ROOT),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "supported": sys.version_info >= (3, 10),
        },
        "packages": packages,
        "data": {
            "root": str(data_root),
            "available": data_root.is_dir(),
            "wav_files": len(list(data_root.rglob("*.wav"))) if data_root.is_dir() else 0,
        },
        "manifest": manifest_report(manifest_path, data_root),
        "huggingface_cache": storage_report(
            resolve_from_project(os.environ.get("HF_HOME", ROOT / ".cache" / "huggingface"))
        ),
    }

    torch = imported.get("torch")
    report["cuda"] = cuda_report(torch) if torch is not None else {"available": False}
    torch_version = packages["torch"].get("version")
    torchaudio_version = packages["torchaudio"].get("version")
    report["torch_torchaudio_match"] = (
        major_minor(str(torch_version)) == major_minor(str(torchaudio_version))
        if torch_version and torchaudio_version
        else False
    )

    transformers = imported.get("transformers")
    try:
        model_class = (
            getattr(transformers, "Wav2Vec2ConformerModel")
            if transformers is not None
            else None
        )
        report["wav2vec2_conformer_class_available"] = model_class is not None
    except Exception as error:  # noqa: BLE001 - report lazy import/binary errors
        report["wav2vec2_conformer_class_available"] = False
        report["wav2vec2_conformer_class_error"] = f"{type(error).__name__}: {error}"
    if args.output_root:
        report["output_storage"] = storage_report(resolve_from_project(args.output_root))

    required_checks = {
        "python_supported": report["python"]["supported"],
        "packages_available": all(item["available"] for item in packages.values()),
        "torch_torchaudio_match": report["torch_torchaudio_match"],
        "cuda_available": report["cuda"]["available"],
        "wav2vec2_conformer_class_available": report[
            "wav2vec2_conformer_class_available"
        ],
        "data_root_available": report["data"]["available"],
        "manifest_available": report["manifest"]["available"],
        "manifest_paths_valid": report["manifest"].get("invalid_paths") == 0,
        "manifest_recordings_available": report["manifest"].get("missing_recordings") == 0,
        "huggingface_cache_writable": report["huggingface_cache"]["writable"],
        "huggingface_cache_space": (
            report["huggingface_cache"]["free_gib"] >= args.min_cache_free_gib
        ),
    }
    if args.output_root:
        required_checks["output_parent_writable"] = report["output_storage"]["writable"]
        required_checks["output_space"] = (
            report["output_storage"]["free_gib"] >= args.min_output_free_gib
        )
    report["required_checks"] = required_checks
    report["status"] = "passed" if all(required_checks.values()) else "failed"

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        failed = [name for name, passed in required_checks.items() if not passed]
        print(f"Preflight failed: {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)
    print("Preflight passed. No model was downloaded and no training was started.")


if __name__ == "__main__":
    main()
