from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath


_WINDOWS_DRIVE_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")
_WSL_UNC_RE = re.compile(r"^\\\\wsl\$\\[^\\]+\\(?P<rest>.*)$", re.IGNORECASE)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def to_platform_path(raw_path: str | os.PathLike[str]) -> Path:
    text = os.fspath(raw_path)
    if os.name != "nt":
        drive_match = _WINDOWS_DRIVE_RE.match(text)
        if drive_match:
            drive = drive_match.group("drive").lower()
            rest = drive_match.group("rest").replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")

        unc_match = _WSL_UNC_RE.match(text)
        if unc_match:
            rest = unc_match.group("rest").replace("\\", "/")
            return Path("/") / rest

    return Path(text)


def resolve_path(
    raw_path: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> Path:
    path = to_platform_path(raw_path).expanduser()
    if path.is_absolute():
        return path

    anchor = Path(base_dir) if base_dir is not None else project_root()
    return (anchor / path).resolve()


def validate_manifest_relative_path(raw_path: str | os.PathLike[str]) -> PurePosixPath:
    """Validate the portable POSIX path stored in a dataset manifest."""
    text = os.fspath(raw_path)
    raw_parts = text.split("/")
    if (
        not text
        or "\\" in text
        or _WINDOWS_DRIVE_RE.match(text)
        or text.startswith("/")
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ValueError(f"Unsafe manifest relative path: {text!r}")
    return PurePosixPath(text)


def resolve_manifest_path(
    data_root: str | os.PathLike[str],
    relative_path: str | os.PathLike[str],
) -> Path:
    """Resolve a portable manifest path under a machine-local dataset root."""
    root = resolve_path(data_root).resolve()
    relative = validate_manifest_relative_path(relative_path)
    candidate = root.joinpath(*relative.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Manifest path escapes data root: {relative.as_posix()!r}") from error
    return candidate


def default_deepship_root() -> str:
    env_value = os.environ.get("DEEPSHIP_DATA_ROOT")
    if env_value:
        return env_value

    candidates = [
        project_root() / "DeepShip",
        project_root() / "Deepship",
        Path.home() / "Transformer" / "DeepShip",
        Path.home() / "Transformer" / "Deepship",
        Path(r"C:\Transformer\DeepShip") if os.name == "nt" else Path("/mnt/c/Transformer/DeepShip"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])
