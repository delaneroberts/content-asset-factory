# caf_app/campaign_generator.py
# caf_app/campaign_generator.py

from __future__ import annotations   # ← MUST be first

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from caf_app.utils import slugify, ensure_campaign_dir, ensure_subdir
from caf_app.text_gen import generate_campaign_texts
from caf_app.storage import campaigns_dir



# ---------------------------------------------------------------------------
# Result container (for internal typing / clarity)
# ---------------------------------------------------------------------------

@dataclass
class CampaignResult:
    campaign_name: str
    slug: str
    campaign_dir: Path
    hero_path: Optional[Path]
    supporting_paths: List[Path]
    zip_path: Path
    text_assets: Dict[str, str]
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _images_base_dir(slug: str) -> Path:
    """
    Base images folder for a campaign: campaigns/<slug>/images

    Even though we are NOT generating images here, we still create
    this folder so that the Image Library can find the expected
    structure if/when the user generates images later.
    """
    return campaigns_dir() / slug / "images"


# ---------------------------------------------------------------------------
# Metadata + ZIP helpers
# ---------------------------------------------------------------------------

def _write_metadata(
    campaign_dir: Path,
    campaign_name: str,
    slug: str,
    campaign_brief: str,
    text_assets: Dict[str, str],
) -> Path:
    """Write a JSON metadata file into the campaign directory."""

    metadata: Dict[str, Any] = {
        "campaign_name": campaign_name,
        "slug": slug,
        "campaign_brief": campaign_brief,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "text_assets_keys": list(text_assets.keys()),
        # Hard-coded because THIS module no longer makes images.
        "has_images": False,
        "hero_image": None,
        "supporting_images": [],
    }

    meta_path = campaign_dir / "campaign.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return meta_path


def _make_zip_export(slug: str, campaign_dir: Path) -> Path:
    """Create a ZIP export of the entire campaign directory."""
    # Archive will be campaigns/<slug>/<slug>.zip
    base_name = campaign_dir / slug
    archive_path = shutil.make_archive(str(base_name), "zip", root_dir=campaign_dir)
    return Path(archive_path)


# ---------------------------------------------------------------------------
# MAIN: generate_campaign()  (TEXT-ONLY, NO IMAGES)
# ---------------------------------------------------------------------------

def generate_campaign(
    campaign_name: str,
    campaign_brief: str,
) -> Dict[str, Any]:
    """Generate campaign assets (TEXT ONLY; no images).

    This function now:
      - Creates / ensures the campaign directory
      - Ensures an empty images/ folder exists
      - Generates text assets via generate_campaign_texts
      - Writes campaign.json metadata with has_images = False
      - Creates a ZIP archive of the campaign folder
      - Returns a simple dict used by the Dashboard
    """

    # ------------------------------------------------------------------
    # 1. Slug + campaign directory
    # ------------------------------------------------------------------
    slug = slugify(campaign_name)

    # Ensure base campaign directory exists.
    campaign_dir = ensure_campaign_dir(slug)

    # Ensure images/ exists (Image Library expects this structure),
    # even though we are not generating any images here.
    images_dir = _images_base_dir(slug)
    images_dir.mkdir(parents=True, exist_ok=True)
    ensure_subdir(campaign_dir, "images")  # redundant but harmless if already created

    # ------------------------------------------------------------------
    # 2. Generate text assets
    # ------------------------------------------------------------------
    text_assets: Dict[str, str] = generate_campaign_texts(
        campaign_name,
        campaign_brief,
    )

    # ------------------------------------------------------------------
    # 3. Metadata + ZIP export
    # ------------------------------------------------------------------
    _write_metadata(
        campaign_dir=campaign_dir,
        campaign_name=campaign_name,
        slug=slug,
        campaign_brief=campaign_brief,
        text_assets=text_assets,
    )

    zip_path = _make_zip_export(slug, campaign_dir)

    # ------------------------------------------------------------------
    # 4. Return dict (used by Dashboard)
    # ------------------------------------------------------------------
    return {
        "campaign_name": campaign_name,
        "slug": slug,
        "campaign_dir": campaign_dir,
        "zip_path": zip_path,
        "text_assets": text_assets,
        # These are kept for compatibility, but will always be empty/None
        "hero_path": None,
        "supporting_paths": [],
    }
