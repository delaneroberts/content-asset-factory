# caf_app/storage.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from slugify import slugify

from .models import Campaign


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """
    Return the project root directory (one level above caf_app).

    Assumes this file lives in <root>/caf_app/storage.py.
    """
    return Path(__file__).resolve().parents[1]


def campaigns_dir() -> Path:
    """
    Directory where campaign JSON files are stored.

    Creates <root>/campaigns if it does not exist.
    """
    root = _project_root()
    directory = root / "campaigns"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _campaign_path(slug: str) -> Path:
    """Return the full path for a campaign JSON file given its slug."""
    return campaigns_dir() / f"{slug}.json"


# ---------------------------------------------------------------------------
# Core campaign storage helpers
# ---------------------------------------------------------------------------


def save_campaign(campaign: Campaign) -> None:
    """
    Write campaign JSON to disk.

    Always refreshes updated_at to now.
    """
    campaign.updated_at = datetime.utcnow()
    path = _campaign_path(campaign.slug)
    with path.open("w", encoding="utf-8") as f:
        json.dump(campaign.model_dump(), f, indent=2, default=str)


def load_campaign(slug: str) -> Optional[Campaign]:
    """
    Load a campaign by slug, or return None if the file does not exist
    or cannot be parsed.
    """
    path = _campaign_path(slug)
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return Campaign(**data)
    except Exception:
        # In a real app you might log this exception.
        return None


def list_campaigns() -> List[Campaign]:
    """
    Return all campaigns (one per JSON file in campaigns/),
    sorted by most recently updated first.
    """
    result: List[Campaign] = []

    for path in campaigns_dir().glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            campaign = Campaign(**data)
            result.append(campaign)
        except Exception:
            # Skip corrupted/unparseable files.
            continue

    result.sort(key=lambda c: c.updated_at, reverse=True)
    return result


# ---------------------------------------------------------------------------
# Creation helpers
# ---------------------------------------------------------------------------


def _ensure_unique_slug(base_slug: str) -> str:
    """
    If a campaign with this slug already exists, append -1, -2, etc.
    until we find a free slug.
    """
    slug = base_slug
    i = 1
    while _campaign_path(slug).exists():
        slug = f"{base_slug}-{i}"
        i += 1
    return slug


def create_campaign(name: str) -> Campaign:
    """
    Create a new campaign with a unique slug and save it.

    Returns the new Campaign instance.
    """
    base_slug = slugify(name) or "campaign"
    slug = _ensure_unique_slug(base_slug)

    campaign = Campaign(
        slug=slug,
        name=name,
    )
    save_campaign(campaign)
    return campaign
