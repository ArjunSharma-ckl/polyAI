from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def find_env_file(start: Optional[Path] = None, *, max_depth: int = 3) -> Optional[Path]:
    """Find a ``.env`` file in the current directory or up to ``max_depth`` parents."""

    current = (start or Path.cwd()).resolve()
    for index, candidate_dir in enumerate([current, *current.parents]):
        if index > max_depth:
            break
        candidate = candidate_dir / ".env"
        if candidate.is_file():
            return candidate
    return None


def _manual_load(path: Path, *, override: bool = False) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if override or key not in os.environ:
            os.environ[key] = value


def load_env_file(path: Optional[str | Path] = None, *, override: bool = False) -> Optional[Path]:
    """Load environment variables from ``path`` or the nearest ``.env`` file.

    Uses ``python-dotenv`` when installed and falls back to a small parser for
    simple ``KEY=value`` files.
    """

    env_path = Path(path).resolve() if path else find_env_file()
    if env_path is None:
        return None
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=override)
    except Exception:
        _manual_load(env_path, override=override)
    return env_path
