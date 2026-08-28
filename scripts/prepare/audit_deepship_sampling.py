from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sampling_audit import (  # noqa: E402
    build_sampling_audit,
    load_csv_rows,
    load_manifest,
    write_sampling_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit current and proposed DeepShip training sampling exposure.",
    )
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--recording-assignments", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clip-duration", type=float, default=20.0)
    parser.add_argument("--epoch-samples", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest, manifest_hash = load_manifest(args.split_manifest)
    inventory, inventory_hash = load_csv_rows(args.inventory)
    assignments, assignments_hash = load_csv_rows(args.recording_assignments)
    audit = build_sampling_audit(
        manifest,
        inventory,
        assignments,
        clip_duration_seconds=args.clip_duration,
        epoch_samples=args.epoch_samples,
    )
    write_sampling_audit(
        audit,
        args.output_dir,
        source_hashes={
            "split_manifest": manifest_hash,
            "inventory": inventory_hash,
            "recording_assignments": assignments_hash,
        },
    )
    redundancy = audit["fixed_context_redundancy"]
    assert isinstance(redundancy, dict)
    overlap = redundancy["adjacent_window_overlap_fraction"]
    assert isinstance(overlap, dict)
    print(
        f"recordings={audit['recordings']} vessels={audit['vessels']} "
        f"median_adjacent_overlap={float(overlap['median']):.1%}"
    )
    print(f"Audit written to: {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
