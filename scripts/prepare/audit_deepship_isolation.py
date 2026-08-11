from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.deepship_audit import (  # noqa: E402
    build_audit_report,
    build_exclusion_rows,
    build_inventory_rows,
    load_experiment_config,
    portable_identity_rows,
    write_audit_outputs,
)
from src.data.deepship_metadata import (  # noqa: E402
    build_manifest_rows,
    load_aliases,
    match_recordings,
    parse_metadata_records,
    scan_metadata_audio,
)
from src.utils.pathing import default_deepship_root  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit DeepShip recordings and vessel identities for isolation experiments.",
    )
    parser.add_argument("--data-root", default=default_deepship_root())
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "isolation_comparison_v1.json"),
    )
    parser.add_argument(
        "--aliases",
        default=str(ROOT / "configs" / "deepship_vessel_aliases.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "protocols" / "isolation_comparison_v1" / "audit"),
    )
    parser.add_argument(
        "--skip-content-hash",
        action="store_true",
        help="Fast diagnostic only; the frozen strict audit requires complete WAV SHA-256 values.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_experiment_config(args.config)
    aliases = load_aliases(args.aliases)
    metadata_records, parse_issues = parse_metadata_records(args.data_root, aliases=aliases)
    audio_records = scan_metadata_audio(args.data_root)
    matches, used_audio, used_metadata = match_recordings(audio_records, metadata_records)
    manifest_rows = build_manifest_rows(audio_records, metadata_records, matches)
    identity_rows = portable_identity_rows(manifest_rows)

    identity_policy = config["identity_policy"]
    assert isinstance(identity_policy, dict)
    allowed_confidence = {
        str(value) for value in identity_policy["allowed_match_confidence"]
    }
    exclusions = build_exclusion_rows(
        identity_rows,
        allowed_confidence=allowed_confidence,
    )
    features = config["features"]
    assert isinstance(features, dict)
    inventory_rows = build_inventory_rows(
        args.data_root,
        manifest_rows,
        clip_duration_seconds=float(features["clip_duration_seconds"]),
        hash_audio=not args.skip_content_hash,
    )
    report = build_audit_report(
        config,
        inventory_rows,
        identity_rows,
        exclusions,
        metadata_record_count=len(metadata_records),
        unmatched_metadata_count=len(metadata_records) - len(used_metadata),
        parse_issues=parse_issues,
    )
    write_audit_outputs(
        args.output_dir,
        inventory_rows,
        identity_rows,
        exclusions,
        report,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit("DeepShip isolation audit failed; inspect identity_audit.json")


if __name__ == "__main__":
    main()
