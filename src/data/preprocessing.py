from __future__ import annotations

from typing import Any


def clean_text(value: Any) -> str:
    """Convert text-like input to a normalized non-null string."""
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return " ".join(text.split())


def normalize_label(value: Any, real_values: list[Any], fake_values: list[Any]) -> int | None:
    """Map a raw label value to 0=Real, 1=Fake, or None when invalid."""
    value_text = str(value).strip().lower()
    real_set = {str(item).strip().lower() for item in real_values}
    fake_set = {str(item).strip().lower() for item in fake_values}

    if value_text in real_set:
        return 0
    if value_text in fake_set:
        return 1
    return None
