# caf_app/storage.py
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from types import SimpleNamespace

from .models import Campaign

# Optional image / EXIF support
try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

try:
    import piexif  # type: ignore
except ImportError:
    piexif = None  # type: ignore


# ---------------------------------------------------------------------------
# Simple built-in slugify (no external dependency)
# ---------------------------------------------------------------------------


def slugify(value: str) -> str:
    """
    Simple slugify implementation.

    Examples:
        "EVO Soda Launch!" -> "evo-soda-launch"
        "Super Dude 2025"  -> "super-dude-2025"
    """
    value = value.lower().strip()
    # replace non-alphanumeric with hyphens
    value = re.sub(r"[^a-z0-9]+", "-", value)
    # collapse multiple hyphens
    value = re.sub(r"-+", "-", value)
    # trim leading/trailing hyphens
    return value.strip("-")


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


def campaign_json_path(slug: str) -> Path:
    """
    Path to the JSON file for a specific campaign.
    """
    return campaigns_dir() / f"{slug}.json"


def campaign_dir(slug: str) -> Path:
    """
    Directory for a specific campaign: campaigns/<slug>/
    """
    cd = campaigns_dir() / slug
    cd.mkdir(parents=True, exist_ok=True)
    return cd


def ensure_campaign_image_folders(slug: str) -> Path:
    """
    Ensure enterprise-style image folders exist for this campaign.

    Structure:
        campaigns/<slug>/
            images/
                legacy/
                generated/
                variants/
                exports/

    Returns the base images/ directory.
    """
    base = campaign_dir(slug) / "images"
    for sub in ["legacy", "generated", "variants", "exports"]:
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def legacy_generated_images_root() -> Path:
    """
    Top-level directory where old images currently live:
        <root>/generated_images
    """
    return _project_root() / "generated_images"


def legacy_campaign_images_dir(slug: str) -> Path:
    """
    Old location for campaign images: generated_images/<slug>/
    """
    return legacy_generated_images_root() / slug


# ---------------------------------------------------------------------------
# Campaign (de)serialization helpers
# ---------------------------------------------------------------------------


def _campaign_to_dict(campaign: Campaign) -> Dict[str, Any]:
    """
    Convert a Campaign object to a serializable dict.

    Tries common patterns (pydantic, dataclass, plain object).
    """
    if hasattr(campaign, "model_dump"):
        return campaign.model_dump()  # type: ignore[attr-defined]
    if hasattr(campaign, "dict"):
        return campaign.dict()  # type: ignore[attr-defined]
    if hasattr(campaign, "__dict__"):
        return dict(campaign.__dict__)
    raise TypeError("Unsupported Campaign object type for serialization")


def _campaign_from_dict(data: Dict[str, Any]) -> Campaign:
    """
    Construct a Campaign from a dict.

    Assumes Campaign(**data) works (dataclass / pydantic / simple class).
    """
    return Campaign(**data)


# ---------------------------------------------------------------------------
# Campaign CRUD
# ---------------------------------------------------------------------------


def list_campaigns() -> List[Campaign]:
    """
    Return a list of all campaigns by reading *.json files under /campaigns.

    Skips any corrupted JSON files instead of crashing the app.
    """
    result: List[Campaign] = []
    for path in campaigns_dir().glob("*.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            result.append(_campaign_from_dict(data))
        except json.JSONDecodeError as e:
            print(f"[storage] Skipping corrupted campaign file {path}: {e}")
            continue
        except Exception as e:
            print(f"[storage] Skipping campaign file {path} due to error: {e}")
            continue
    return result

def load_campaign(slug: str) -> SimpleNamespace:
    """
    Load a campaign by slug from the canonical structure:

        campaigns/<slug>/campaign.json

    If the file doesn't exist yet, fabricate a minimal campaign object so the
    UI can still open and let the user fill things in.
    """
    root = campaigns_dir()
    base_dir = root / slug
    base_dir.mkdir(parents=True, exist_ok=True)

    json_path = base_dir / "campaign.json"

    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # If somehow it's corrupted, start fresh rather than crashing
            data = {}
    else:
        data = {}

    # ---- Minimal identity ----
    data.setdefault("slug", slug)
    data.setdefault("name", slug)

    # ---- Common fields used in Brief Editor ----
    data.setdefault("product_name", data.get("name"))
    data.setdefault("description", "")
    data.setdefault("audience", "")
    data.setdefault("tone", "")
    data.setdefault("notes", "")

    # ---- Fields expected by Text Library / other pages ----
    # Text Library expects a list of TextAsset-like objects
    if "text_assets" not in data or data["text_assets"] is None:
        data["text_assets"] = []

    # Where text files live (if/when you wire them)
    if "copy_files" not in data or data["copy_files"] is None:
        data["copy_files"] = {}

    # Image-related structures some pages expect; safe empty defaults
    if "images" not in data or data["images"] is None:
        data["images"] = {}

    if "image_assets" not in data or data["image_assets"] is None:
        data["image_assets"] = []

    # Concept Builder / Review & Export expect a list of concepts
    if "concepts" not in data or data["concepts"] is None:
        data["concepts"] = []

    return SimpleNamespace(**data)


def save_campaign(campaign: SimpleNamespace) -> None:
    """
    Save a campaign to:

        campaigns/<slug>/campaign.json

    Works with the SimpleNamespace returned by load_campaign and any object
    with similar attributes.
    """
    root = campaigns_dir()

    # Turn whatever we got into a plain dict
    data = vars(campaign).copy()

    slug = (
        data.get("slug")
        or data.get("name")
        or data.get("product_name")
        or "unnamed-campaign"
    )
    slug = str(slug)
    data["slug"] = slug

    base_dir = root / slug
    base_dir.mkdir(parents=True, exist_ok=True)

    json_path = base_dir / "campaign.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)



def create_campaign(name: str, brief: str) -> Campaign:
    """
    Helper to create and save a new campaign from a name + brief.
    Ensures unique slug.
    """
    base_slug = slugify(name)
    slug = base_slug
    i = 1
    # ensure uniqueness
    while campaign_json_path(slug).exists():
        slug = f"{base_slug}-{i}"
        i += 1

    now = datetime.utcnow().isoformat()
    data: Dict[str, Any] = {
        "name": name,
        "slug": slug,
        "brief": brief,
        "created_at": now,
        "updated_at": now,
    }
    campaign = _campaign_from_dict(data)
    return save_campaign(campaign)


def delete_campaign(slug: str) -> None:
    """
    Delete a campaign and all of its stored assets.

    - Removes campaigns/<slug>/ directory (images, exports, etc.)
    - Removes campaigns/<slug>.json metadata file
    - Removes legacy generated_images/<slug>/ folder if present
    """
    # Remove JSON metadata
    json_path = campaign_json_path(slug)
    if json_path.exists():
        json_path.unlink()

    # Remove campaign directory
    cd = campaign_dir(slug)
    if cd.exists():
        shutil.rmtree(cd)

    # Remove legacy images, if any
    legacy_dir = legacy_campaign_images_dir(slug)
    if legacy_dir.exists():
        shutil.rmtree(legacy_dir)


# ---------------------------------------------------------------------------
# Image naming & metadata helpers
# ---------------------------------------------------------------------------


def build_image_filename(slug: str, variant: str, seq: int, ext: str = "png") -> str:
    """
    Enterprise-style filename:

        <campaignSlug>_<variant>_<sequence>.<ext>
        e.g. evo-soda-launch_hero_01.png
    """
    return f"{slug}_{variant}_{seq:02d}.{ext.lstrip('.')}"


def write_image_metadata_json(image_path: Path, metadata: Dict[str, Any]) -> Path:
    """
    Write a JSON sidecar file next to the image with metadata.

    Example:
        image.png  -> image.json
    """
    json_path = image_path.with_suffix(".json")
    meta_copy = dict(metadata)
    if "created_at" not in meta_copy:
        meta_copy["created_at"] = datetime.utcnow().isoformat()
    with open(json_path, "w") as f:
        json.dump(meta_copy, f, indent=2, default=str)
    return json_path


def embed_metadata_in_image(image_path: Path, metadata: Dict[str, Any]) -> None:
    """
    Optionally embed basic metadata (e.g., alt_text) into the image itself.

    If Pillow or piexif are not installed, this is a no-op.
    """
    if Image is None or piexif is None:
        return

    alt_text = metadata.get("alt_text")
    if not alt_text:
        return

    img = Image.open(image_path)
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}}
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = str(alt_text)
    exif_bytes = piexif.dump(exif_dict)
    img.save(image_path, exif=exif_bytes)


def save_generated_image_bytes(
    slug: str,
    filename: str,
    image_bytes: bytes,
    metadata: Optional[Dict[str, Any]] = None,
    subfolder: str = "generated",
) -> Path:
    """
    Save generated image bytes into the new enterprise folder structure and
    optionally write metadata.

    - slug: campaign slug
    - filename: already-built filename (e.g. from build_image_filename)
    - metadata: dict with fields like {campaign, variant, sequence, engine, alt_text,...}
    - subfolder: 'generated', 'variants', etc.
    """
    images_base = ensure_campaign_image_folders(slug)
    dest_dir = images_base / subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    with open(dest_path, "wb") as f:
        f.write(image_bytes)

    if metadata:
        write_image_metadata_json(dest_path, metadata)
        embed_metadata_in_image(dest_path, metadata)

    return dest_path


def list_all_images_for_campaign(slug: str) -> List[Path]:
    """
    List all known images for a campaign, combining:

        campaigns/<slug>/images/legacy/
        campaigns/<slug>/images/generated/
        campaigns/<slug>/images/variants/
        generated_images/<slug>/    (old legacy location)

    Returns unique, sorted Paths.
    """
    images_base = ensure_campaign_image_folders(slug)
    folders = [
        images_base / "legacy",
        images_base / "generated",
        images_base / "variants",
    ]

    legacy_root = legacy_campaign_images_dir(slug)
    if legacy_root.exists():
        folders.append(legacy_root)

    seen = set()
    result: List[Path] = []

    for folder in folders:
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            if f in seen:
                continue
            seen.add(f)
            result.append(f)

    return sorted(result)


# ---------------------------------------------------------------------------
# Migration helpers (non-destructive)
# ---------------------------------------------------------------------------


def migrate_legacy_images_for_campaign(slug: str) -> None:
    """
    Copy images from generated_images/<slug>/ into
    campaigns/<slug>/images/legacy/ (non-destructive).

    Existing files in the destination are NOT overwritten.
    """
    src = legacy_campaign_images_dir(slug)
    if not src.exists():
        return

    base = ensure_campaign_image_folders(slug)
    dest = base / "legacy"

    for f in src.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue

        target = dest / f.name
        if target.exists():
            continue  # already migrated

        shutil.copy2(f, target)


def migrate_all_campaigns() -> None:
    """
    Migrate all existing legacy images by scanning generated_images/
    and copying each campaign folder into campaigns/<slug>/images/legacy/.

    This is non-destructive: nothing in generated_images/ is deleted.
    """
    root = legacy_generated_images_root()
    if not root.exists():
        return

    for child in root.iterdir():
        if child.is_dir():
            slug = child.name
            print(f"Migrating legacy images for slug: {slug}")
            migrate_legacy_images_for_campaign(slug)
