from __future__ import annotations

import re
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGNS_ROOT = PROJECT_ROOT / "campaigns"


def slugify(value: str) -> str:
    """
    Convert a string into a filesystem-safe slug.
    """
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "campaign"


def ensure_campaign_dir(slug: str, base_dir: Path | None = None) -> Path:
    """
    Create (if needed) and return a campaign directory.
    Structure: <base_dir>/<slug>/
    """
    base = base_dir or DEFAULT_CAMPAIGNS_ROOT
    base.mkdir(parents=True, exist_ok=True)
    campaign_dir = base / slug
    campaign_dir.mkdir(parents=True, exist_ok=True)
    return campaign_dir


def ensure_subdir(parent: Path, name: str) -> Path:
    """
    Ensure a named subdirectory under parent.
    """
    subdir = parent / name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir
