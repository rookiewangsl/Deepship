from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.utils.pathing import resolve_manifest_path, validate_manifest_relative_path


class ManifestPathTests(unittest.TestCase):
    def test_preserves_portable_posix_path(self) -> None:
        path = validate_manifest_relative_path("Cargo/20171104-1/1.wav")

        self.assertEqual(path.as_posix(), "Cargo/20171104-1/1.wav")

    def test_resolves_under_machine_local_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = resolve_manifest_path(temp_dir, "Tank/example.wav")

            self.assertEqual(result, Path(temp_dir).resolve() / "Tank" / "example.wav")

    def test_rejects_parent_escape(self) -> None:
        with self.assertRaises(ValueError):
            resolve_manifest_path("/tmp/data", "../secret.wav")

    def test_rejects_windows_absolute_path_in_manifest(self) -> None:
        with self.assertRaises(ValueError):
            validate_manifest_relative_path(r"E:\\DeepShip\\Cargo\\example.wav")

    def test_rejects_backslash_relative_path(self) -> None:
        with self.assertRaises(ValueError):
            validate_manifest_relative_path(r"Cargo\\example.wav")


if __name__ == "__main__":
    unittest.main()
