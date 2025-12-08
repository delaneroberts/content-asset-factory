from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from caf_app.utils import slugify, ensure_campaign_dir, ensure_subdir
from caf_app.text_gen import generate_campaign_texts
from caf_app.image_gen import (
    generate_evo_hero_image,
    generate_evo_support_image,
    ImageProvider,
)


@dataclass
class CampaignResult:
    campaign_name: str
    slug: str
    campaign_dir: Path
    hero_path: Path
    supporting_paths: list[Path]
    zip_path: Path
    text_assets: Dict[str, str]
    metadata: Dict[str, Any]


def generate_campaign(
    campaign_name: str,
    campaign_brief: str,
    base_output_dir: Path | None = None,
    image_provider: ImageProvider = ImageProvider.OPENAI,
) -> Dict[str, Any]:
    """
    High-level orchestration for generating a campaign.

    - Slugify campaign name
    - Create campaign directory under campaigns/<slug>/
    - Generate text assets (tagline, slogans, etc.)
    - Generate images (hero + supporting) via selected provider
    - Write all copy to disk
    - Save prompts used
    - Save metadata JSON
    - Create ZIP archive of campaign folder

    Returns a dict suitable for the Streamlit UI.
    """

    # ---------- Slug + folder ----------
    slug = slugify(campaign_name or "campaign")
    campaign_dir = ensure_campaign_dir(slug, base_output_dir)

    # ---------- Text generation ----------
    text_assets = generate_campaign_texts(
        campaign_name=campaign_name,
        campaign_brief=campaign_brief,
    )

    # ---------- Image generation ----------
    # We generate into generated_images/ via image_gen.py,
    # then copy into this campaign's images/ subfolder.

    images_dir = ensure_subdir(campaign_dir, "images")

    # Hero image
    hero_tmp = generate_evo_hero_image(
        campaign_brief,
        filename=f"{slug}_hero.png",
    )
    hero_path: Path = images_dir / hero_tmp.name
    shutil.copy(hero_tmp, hero_path)

    # Supporting images (3 variants)
    supporting_paths: list[Path] = []
    for i in range(1, 4):
        supp_tmp = generate_evo_support_image(
            campaign_brief,
            filename=f"{slug}_support_{i}.png",
        )
        supp_final = images_dir / supp_tmp.name
        shutil.copy(supp_tmp, supp_final)
        supporting_paths.append(supp_final)

    # ---------- Write copy files ----------
    copy_dir = ensure_subdir(campaign_dir, "copy")

    (copy_dir / "tagline.txt").write_text(
        text_assets["tagline"], encoding="utf-8"
    )
    (copy_dir / "slogans.txt").write_text(
        text_assets["slogans"], encoding="utf-8"
    )
    (copy_dir / "value_prop.txt").write_text(
        text_assets["value_prop"], encoding="utf-8"
    )
    (copy_dir / "ctas.txt").write_text(
        text_assets["ctas"], encoding="utf-8"
    )
    (copy_dir / "social_posts.txt").write_text(
        text_assets["social_posts"], encoding="utf-8"
    )
    (copy_dir / "campaign_summary.txt").write_text(
        text_assets["summary"], encoding="utf-8"
    )

    # Prompts file (all prompts used for LLM)
    prompts_path = copy_dir / "generation_prompts.txt"
    prompts_path.write_text(text_assets["_prompts"], encoding="utf-8")

    # ---------- Metadata ----------
    metadata: Dict[str, Any] = {
        "campaign_name": campaign_name,
        "slug": slug,
        "brief": campaign_brief,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "image_provider": image_provider.value if isinstance(image_provider, ImageProvider) else str(image_provider),
        "images": {
            "hero": str(hero_path),
            "supporting": [str(p) for p in supporting_paths],
        },
        "copy_files": {
            "tagline": str(copy_dir / "tagline.txt"),
            "slogans": str(copy_dir / "slogans.txt"),
            "value_prop": str(copy_dir / "value_prop.txt"),
            "ctas": str(copy_dir / "ctas.txt"),
            "social_posts": str(copy_dir / "social_posts.txt"),
            "summary": str(copy_dir / "campaign_summary.txt"),
            "prompts": str(prompts_path),
        },
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
    result: Dict[str, Any] = {
        "campaign_name": campaign_name,
        "slug": slug,
        "campaign_dir": campaign_dir,
        "hero_path": hero_path,
        "supporting_paths": supporting_paths,
        "zip_path": zip_path,
        "text_assets": text_assets,
        "metadata": metadata,
    }

    return result

