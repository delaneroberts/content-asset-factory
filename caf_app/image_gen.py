from __future__ import annotations

import base64
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

try:
    # modern OpenAI SDK
    from openai import OpenAI
except ImportError:
    OpenAI = None


class ImageProvider(str, Enum):
    OPENAI = "openai"
    PLACEHOLDER = "placeholder"


def _project_root() -> Path:
    # assumes this file is at <root>/caf_app/image_gen.py
    return Path(__file__).resolve().parents[1]


def _images_dir() -> Path:
    d = _project_root() / "generated_images"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
    images_dir = _images_dir()
    out_path = images_dir / filename

    try:
        if provider == ImageProvider.OPENAI:
            return _generate_openai_image(brief, variant_label, out_path)
        else:
            return _generate_placeholder_image(variant_label, out_path)
    except Exception as exc:
        print(f"[image_gen] Error using provider {provider}: {exc}")
        return _generate_placeholder_image(variant_label, out_path)


# ---------- OpenAI provider (real API) ----------

def _generate_openai_image(
    brief: str,
    variant_label: str,
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

    print(f"[image_gen] Saved OpenAI image to {out_path}")
    return out_path


# ---------- Placeholder fallback ----------

def _generate_placeholder_image(label: str, out_path: Path) -> Path:
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
    print(f"[image_gen] Saved placeholder image to {out_path}")
    return out_path
