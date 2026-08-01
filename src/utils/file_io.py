from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory for a file path if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary as indented JSON."""
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
