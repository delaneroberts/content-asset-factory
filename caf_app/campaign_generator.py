# caf_app/campaign_generator.py

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from caf_app.utils import slugify, ensure_campaign_dir, ensure_subdir
from caf_app.text_gen import generate_campaign_texts

# NEW imports: multi-engine image generator
from caf_app.image_gen import generate_image, save_png
from caf_app.storage import campaigns_dir


# ---------------------------------------------------------------------------
# Result container (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class CampaignResult:
    campaign_name: str
    slug: str
    campaign_dir: Path
    hero_path: Path
    supporting_paths: List[Path]
    zip_path: Path
    text_assets: Dict[str, str]
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Image directory helpers (MATCH the Image Library)
# ---------------------------------------------------------------------------

def _images_base_dir(slug: str) -> Path:
    """Base images folder for a campaign."""
    return campaigns_dir() / slug / "images"


def _generated_dir(slug: str) -> Path:
    """Generated images go here: campaigns/<slug>/images/generated"""
    d = _images_base_dir(slug) / "generated"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Multi-engine image generation for hero + variants
# ---------------------------------------------------------------------------

def _generate_hero_image(
    slug: str,
    brief: str,
    engine: str = "auto",
    size: str = "1024x1024",
) -> Path:
    """Generate a hero image using the new multi-engine adapter."""

    image_bytes, engine_used = generate_image(
        prompt=brief,
        engine=engine,
        size=size,
    )

    out_dir = _generated_dir(slug)
    filename = f"{slug}_hero.png"
    out_path = out_dir / filename

    save_png(image_bytes, out_path)

    print(f"[CAF] Hero generated with engine: {engine_used}")

    return out_path


def _generate_support_images(
    slug: str,
    brief: str,
    num_images: int = 3,
    engine: str = "auto",
    size: str = "1024x1024",
) -> List[Path]:
    """Generate supporting images using multi-engine adapter."""

    out_dir = _generated_dir(slug)
    results: List[Path] = []

    for i in range(num_images):
        image_bytes, engine_used = generate_image(
            prompt=brief,
            engine=engine,
            size=size,
        )

        ts = int(datetime.utcnow().timestamp() * 1000)
        filename = f"{slug}_support_{i}_{ts}.png"
        out_path = out_dir / filename

        save_png(image_bytes, out_path)
        results.append(out_path)

        print(f"[CAF] Support image {i+1} generated with engine: {engine_used}")

    return results


# ---------------------------------------------------------------------------
# MAIN: generate_campaign()
# ---------------------------------------------------------------------------

def generate_campaign(
    campaign_name: str,
    campaign_brief: str,
    base_output_dir: Path | None = None,
    image_engine: str = "auto",       # << NEW: user can choose openai / nanobanana / auto
    num_supporting: int = 3,          # << NEW: easy to configure
) -> Dict[str, Any]:
    """
    High-level orchestration for generating a campaign.

    - Create campaign directory under campaigns/<slug>/
    - Generate text assets
    - Generate hero + supporting images via multi-engine generator
    - Save everything the UI needs
    """

    # ---------- Slug + folder ----------
    slug = slugify(campaign_name or "campaign")
    campaign_dir = ensure_campaign_dir(slug, base_output_dir)

    # ---------- Prepare image folders ----------
    images_dir = ensure_subdir(campaign_dir, "images")
    # (Sub-directory 'generated' created automatically in generators)

    # ---------- Text generation ----------
    text_assets = generate_campaign_texts(
        campaign_name=campaign_name,
        campaign_brief=campaign_brief,
    )

    # ---------- Image generation (NEW multi-engine) ----------
    hero_path = _generate_hero_image(
        slug=slug,
        brief=campaign_brief,
        engine=image_engine,
    )

    supporting_paths = _generate_support_images(
        slug=slug,
        brief=campaign_brief,
        num_images=num_supporting,
        engine=image_engine,
    )

    # ---------- Write copy files ----------
    copy_dir = ensure_subdir(campaign_dir, "copy")

    (copy_dir / "tagline.txt").write_text(text_assets["tagline"], encoding="utf-8")
    (copy_dir / "slogans.txt").write_text(text_assets["slogans"], encoding="utf-8")
    (copy_dir / "value_prop.txt").write_text(text_assets["value_prop"], encoding="utf-8")
    (copy_dir / "ctas.txt").write_text(text_assets["ctas"], encoding="utf-8")
    (copy_dir / "social_posts.txt").write_text(text_assets["social_posts"], encoding="utf-8")
    (copy_dir / "campaign_summary.txt").write_text(text_assets["summary"], encoding="utf-8")
    (copy_dir / "generation_prompts.txt").write_text(text_assets["_prompts"], encoding="utf-8")

    # ---------- Metadata ----------
    metadata: Dict[str, Any] = {
        "campaign_name": campaign_name,
        "slug": slug,
        "brief": campaign_brief,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "image_engine": image_engine,                     # << NEW
        "images": {
            "hero": str(hero_path),
            "supporting": [str(p) for p in supporting_paths],
        },
        "copy_files": {name: str(copy_dir / f"{name}.txt")
                       for name in ["tagline", "slogans", "value_prop",
                                     "ctas", "social_posts", "campaign_summary"]},
        "prompts_file": str(copy_dir / "generation_prompts.txt"),
    }

    metadata_path = campaign_dir / "campaign_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # ---------- ZIP export ----------
    zip_path = campaign_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()

    shutil.make_archive(
        base_name=str(campaign_dir),
        format="zip",
        root_dir=str(campaign_dir.parent),
        base_dir=campaign_dir.name,
    )

    # ---------- Return structure for UI ----------
    return {
        "campaign_name": campaign_name,
        "slug": slug,
        "campaign_dir": campaign_dir,
        "hero_path": hero_path,
        "supporting_paths": supporting_paths,
        "zip_path": zip_path,
        "text_assets": text_assets,
        "metadata": metadata,
    }
