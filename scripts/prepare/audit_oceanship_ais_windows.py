from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


UTC_FORMAT = "%Y%m%dT%H%M%S.%fZ"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decode position reports around Oceanship-FG candidate timestamps and "
            "audit whether the labelled MMSI is actually near the ONC hydrophone."
        )
    )
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--device-code", required=True)
    parser.add_argument("--receiver-latitude", required=True, type=float)
    parser.add_argument("--receiver-longitude", required=True, type=float)
    parser.add_argument("--window-seconds", default=300.0, type=float)
    parser.add_argument("--radius-km", nargs="+", default=[1.0, 5.0, 11.0], type=float)
    parser.add_argument("--output-json", required=True)
    return parser


def parse_utc(value: str) -> datetime:
    return datetime.strptime(value, UTC_FORMAT).replace(tzinfo=timezone.utc)


def haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius_km = 6371.0088
    phi_a = math.radians(lat_a)
    phi_b = math.radians(lat_b)
    delta_phi = math.radians(lat_b - lat_a)
    delta_lambda = math.radians(lon_b - lon_a)
    value = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(value))


def read_candidates(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def archive_path(
    archive_dir: Path,
    device_code: str,
    event_timestamp: datetime,
) -> Path:
    day = event_timestamp.strftime("%Y%m%d")
    return archive_dir / f"{device_code}_{day}T000000.000Z.txt"


def decode_window(
    path: Path,
    event_timestamp: datetime,
    *,
    window_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    try:
        from pyais import decode
    except ImportError as error:
        raise RuntimeError(
            "pyais is unavailable; install the isolated requirements-onc.txt environment"
        ) from error

    reports: list[dict[str, Any]] = []
    counters = {
        "lines_in_window": 0,
        "ais_sentences_in_window": 0,
        "decode_errors": 0,
        "position_reports": 0,
    }
    with path.open("rb") as handle:
        for raw_line in handle:
            parts = raw_line.rstrip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            try:
                line_timestamp = parse_utc(parts[0].decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                continue
            delta_seconds = (line_timestamp - event_timestamp).total_seconds()
            if abs(delta_seconds) > window_seconds:
                continue
            counters["lines_in_window"] += 1
            sentence = parts[1]
            if not sentence.startswith((b"!AIVDM", b"!AIVDO")):
                continue
            counters["ais_sentences_in_window"] += 1
            try:
                message = decode(sentence).asdict()
            except Exception:
                counters["decode_errors"] += 1
                continue
            latitude = message.get("lat")
            longitude = message.get("lon")
            mmsi = message.get("mmsi")
            if latitude is None or longitude is None or mmsi is None:
                continue
            latitude = float(latitude)
            longitude = float(longitude)
            if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
                continue
            counters["position_reports"] += 1
            reports.append(
                {
                    "timestamp_utc": line_timestamp.strftime(UTC_FORMAT),
                    "delta_seconds": delta_seconds,
                    "mmsi": str(mmsi),
                    "message_type": int(message["msg_type"]),
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )
    return reports, counters


def summarize_candidate(
    candidate: dict[str, str],
    path: Path,
    reports: list[dict[str, Any]],
    counters: dict[str, int],
    *,
    receiver_latitude: float,
    receiver_longitude: float,
    radii_km: list[float],
) -> dict[str, Any]:
    for report in reports:
        report["distance_to_receiver_km"] = haversine_km(
            receiver_latitude,
            receiver_longitude,
            report["latitude"],
            report["longitude"],
        )

    target_reports = [report for report in reports if report["mmsi"] == candidate["mmsi"]]
    closest_in_time = min(target_reports, key=lambda report: abs(report["delta_seconds"]), default=None)
    unique_mmsi = {report["mmsi"] for report in reports}
    nearest_by_mmsi: dict[str, float] = {}
    for report in reports:
        mmsi = report["mmsi"]
        nearest_by_mmsi[mmsi] = min(
            nearest_by_mmsi.get(mmsi, math.inf),
            report["distance_to_receiver_km"],
        )

    return {
        "label": candidate["label"],
        "target_mmsi": candidate["mmsi"],
        "event_timestamp_utc": candidate["event_timestamp_utc"],
        "archive_file": path.name,
        "archive_file_exists": True,
        **counters,
        "unique_position_mmsis": len(unique_mmsi),
        "unique_mmsis_within_radius": {
            str(radius): sum(distance <= radius for distance in nearest_by_mmsi.values())
            for radius in radii_km
        },
        "target_position_reports": len(target_reports),
        "target_seen": bool(target_reports),
        "target_closest_in_time": closest_in_time,
    }


def main() -> None:
    args = build_parser().parse_args()
    archive_dir = Path(args.archive_dir)
    candidates = read_candidates(args.candidate_csv)
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        event_timestamp = parse_utc(candidate["event_timestamp_utc"])
        path = archive_path(archive_dir, args.device_code, event_timestamp)
        if not path.is_file():
            results.append(
                {
                    "label": candidate["label"],
                    "target_mmsi": candidate["mmsi"],
                    "event_timestamp_utc": candidate["event_timestamp_utc"],
                    "archive_file": path.name,
                    "archive_file_exists": False,
                }
            )
            continue
        reports, counters = decode_window(
            path,
            event_timestamp,
            window_seconds=args.window_seconds,
        )
        results.append(
            summarize_candidate(
                candidate,
                path,
                reports,
                counters,
                receiver_latitude=args.receiver_latitude,
                receiver_longitude=args.receiver_longitude,
                radii_km=args.radius_km,
            )
        )

    available = [result for result in results if result["archive_file_exists"]]
    payload = {
        "status": "ais_window_audit_complete",
        "device_code": args.device_code,
        "receiver_latitude": args.receiver_latitude,
        "receiver_longitude": args.receiver_longitude,
        "window_seconds": args.window_seconds,
        "radius_km": args.radius_km,
        "candidate_count": len(results),
        "available_archive_count": len(available),
        "target_seen_count": sum(bool(result.get("target_seen")) for result in available),
        "results": results,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        f"Audited {len(available)}/{len(results)} available AIS windows; "
        f"target MMSI seen in {payload['target_seen_count']}."
    )


if __name__ == "__main__":
    main()
