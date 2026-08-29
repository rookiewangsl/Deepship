from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.oceanship_onc import (  # noqa: E402
    DEFAULT_DEVICE_CODES,
    query_archive_candidates,
    read_probe_candidates,
    redact_onc_error,
    write_archive_index,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Query the ONC archive for files surrounding a small Oceanship-FG candidate "
            "list. This command only writes a filename index and never downloads data."
        )
    )
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device-codes", nargs="+", default=list(DEFAULT_DEVICE_CODES))
    parser.add_argument("--extension", default="wav")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not os.environ.get("ONC_TOKEN"):
        raise SystemExit(
            "ONC_TOKEN is not set. Use a personal Oceans 3.0 token; do not commit it to the project."
        )
    try:
        from onc import ONC
    except ImportError as error:
        raise SystemExit(
            "The optional 'onc' client is unavailable. Install requirements-onc.txt "
            "in a separate preparation environment."
        ) from error

    candidates = read_probe_candidates(args.candidate_csv)
    try:
        client = ONC(showWarning=True)
        rows = query_archive_candidates(
            candidates,
            client,
            device_codes=args.device_codes,
            extension=args.extension,
        )
    except Exception as error:
        raise SystemExit(f"ONC archive query failed: {redact_onc_error(error)}") from None
    write_archive_index(args.output_csv, rows)
    print(
        f"Queried {len(candidates)} events; indexed {len(rows)} matching "
        f".{args.extension} archive files."
    )


if __name__ == "__main__":
    main()
