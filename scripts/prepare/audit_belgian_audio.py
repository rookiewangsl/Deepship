from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.belgian_ais import audit_audio_files, record_from_dict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit every audio file used by Belgian development.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--protocol-root",
        default=str(ROOT / "protocols" / "belgian_attention_v1"),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_root = Path(args.protocol_root).expanduser().resolve()
    manifest = json.loads(
        (protocol_root / "fold1" / "split_manifest.json").read_text(encoding="utf-8")
    )
    records = [record_from_dict(row) for row in manifest["records"]]
    report = audit_audio_files(records, data_root=args.data_root)
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite Belgian audio audit: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

