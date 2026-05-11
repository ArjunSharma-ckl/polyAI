from __future__ import annotations

import json
import sys
from typing import Any


def debug_log(enabled: bool, label: str, payload: Any) -> None:
    """Print debug payloads to stderr when debug mode is enabled."""

    if not enabled:
        return
    if isinstance(payload, (dict, list)):
        rendered = json.dumps(payload, indent=2, default=str)
    else:
        rendered = str(payload)
    print(f"[polyai] {label}: {rendered}", file=sys.stderr)
