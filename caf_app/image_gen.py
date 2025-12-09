from __future__ import annotations

import base64
import os
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

from PIL import Image, ImageDraw

try:
    # modern OpenAI SDK
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Use the new storage helpers so images live under campaigns/<slug>/images/...
from .storage import (
    ensure_campaign_image_folders,
    write_image_metadata_json,
    embed_metadata_in_image,
)


class ImageProvider(str, Enum):
    OPENAI = "openai"
    PLACEHOLDER = "placeholder"


# ---------- Internal helpers ----------


def _infer_slug_from_filename(filename: str) -> str:
    """
    Infer a campaign slug from a filename like 'evo-product-launch-1_hero.png'
    -> 'evo-product-launch-1'.
    If no underscore is present, uses the whole stem as slug.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    return parts[0] if parts else stem


def _images_dir_for_slug(slug: str) -> Path:
    """
    Return the directory where generated images for this campaign should live:

        campaigns/<slug>/images/generated/
    """
    base = ensure_campaign_image_folders(slug)  # campaigns/<slug>/images
    d = base / "generated"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_basic_metadata(
    slug: str,
    variant_label: str,
    brief: str,
    out_path: Path,
    engine_name: str,
) -> None:
    """
    Write a JSON sidecar and (optionally) embed alt_text into the image.
    This is best-effort; failures are logged but not raised.
    """
    metadata: Dict[str, Any] = {
        "campaign": slug,
        "variant": variant_label,
        "engine": engine_name,
        "brief": brief,
        "alt_text": f"{variant_label} image for campaign '{slug}'",
    }
    try:
        write_image_metadata_json(out_path, metadata)
        embed_metadata_in_image(out_path, metadata)
    except Exception as exc:  # noqa: BLE001
        print(f"[image_gen] Warning: could not write metadata for {out_path}: {exc}")


# ---------- Public API used by the app ----------


def generate_evo_hero_image(brief: str, filename: str) -> Path:
    return _generate_evo_image(brief, "hero", filename)


def generate_evo_support_image(brief: str, filename: str) -> Path:
    return _generate_evo_image(brief, "support", filename)


def generate_evo_logo_image(brief: str, filename: str) -> Path:
    return _generate_evo_image(brief, "logo", filename)


def _generate_evo_image(
    brief: str,
    variant_label: str,
    filename: str,
    provider: ImageProvider = ImageProvider.OPENAI,
) -> Path:
    """
    Main entrypoint: chooses provider, routes to OpenAI or placeholder.

    NOTE: We keep the old signature (brief + filename) for compatibility,
    but now infer the campaign slug from the filename and save into:

        campaigns/<slug>/images/generated/<filename>
    """
    slug = _infer_slug_from_filename(filename)
    images_dir = _images_dir_for_slug(slug)
    out_path = images_dir / filename

    try:
        if provider == ImageProvider.OPENAI:
            return _generate_openai_image(brief, variant_label, slug, out_path)
        else:
            return _generate_placeholder_image(variant_label, slug, out_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[image_gen] Error using provider {provider}: {exc}")
        return _generate_placeholder_image(variant_label, slug, out_path)


# ---------- OpenAI provider (real API) ----------


def _generate_openai_image(
    brief: str,
    variant_label: str,
    slug: str,
    out_path: Path,
) -> Path:
    """
    Calls OpenAI gpt-image-1 and writes a PNG to out_path.
    Requires OPENAI_API_KEY in the environment.
    """
    if OpenAI is None:
        raise RuntimeError(
            "OpenAI SDK not installed. Run `pip install --upgrade openai`."
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")

    client = OpenAI(api_key=api_key)

    prompt = (
        f"{brief.strip()}\n\n"
        f"Image type: {variant_label} asset for a marketing campaign."
    )

    print(f"[image_gen] Calling OpenAI for {variant_label} -> {out_path}")

    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=1,
        size="1024x1024",
        quality="high",
        output_format="png",
    )

    b64_data = result.data[0].b64_json
    image_bytes = base64.b64decode(b64_data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(image_bytes)

    _write_basic_metadata(slug, variant_label, brief, out_path, "openai-gpt-image-1")

    print(f"[image_gen] Saved OpenAI image to {out_path}")
    return out_path


# ---------- Placeholder fallback ----------


def _generate_placeholder_image(
    label: str,
    slug: str,
    out_path: Path,
) -> Path:
    img = Image.new("RGB", (1024, 1024), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)

    text = label.upper()
    x0, y0, x1, y1 = draw.textbbox((0, 0), text)
    text_w = x1 - x0
    text_h = y1 - y0

    draw.text(
        ((1024 - text_w) / 2, (1024 - text_h) / 2),
        text,
        fill=(0, 0, 0),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    _write_basic_metadata(slug, label, f"Placeholder {label} asset", out_path, "placeholder")

    print(f"[image_gen] Saved placeholder image to {out_path}")
    return out_path
