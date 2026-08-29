from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import unittest

from scripts.prepare.audit_oceanship_ais_windows import archive_path, haversine_km


class OceanshipAISWindowTests(unittest.TestCase):
    def test_archive_path_uses_event_utc_day(self) -> None:
        event = datetime(2020, 8, 20, 17, 41, 52, tzinfo=timezone.utc)
        result = archive_path(Path("/tmp/archive"), "AISDEVICE", event)
        self.assertEqual(result.name, "AISDEVICE_20200820T000000.000Z.txt")

    def test_haversine_is_zero_for_identical_point(self) -> None:
        self.assertEqual(haversine_km(50.020867, -125.235367, 50.020867, -125.235367), 0.0)

    def test_haversine_matches_known_one_degree_latitude_scale(self) -> None:
        self.assertAlmostEqual(haversine_km(50.0, -125.0, 51.0, -125.0), 111.195, places=3)


if __name__ == "__main__":
    unittest.main()
