from __future__ import annotations
from pathlib import Path
from enum import Enum
from typing import Tuple, List
from PIL import Image, ImageDraw


# -----------------------------
#  Provider Enum
# -----------------------------
class ImageProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    PLACEHOLDER = "placeholder"


# -----------------------------
#  Public API
# -----------------------------

def generate_campaign_images(
    brief: str,
    slug: str,
    campaign_dir: Path,
    provider: ImageProvider = ImageProvider.OPENAI,
) -> dict:
    """
    Generate:
      - 1 hero image
      - 3 supporting images
    Return:
      { "hero_path": Path, "supporting_paths": [Path, Path, Path] }
    """

    images_dir = campaign_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if provider == ImageProvider.OPENAI:
        hero_path, supporting_paths = _generate_openai_images(
            brief, slug, images_dir
        )

    elif provider == ImageProvider.GEMINI:
        hero_path, supporting_paths = _generate_gemini_images(
            brief, slug, images_dir
        )

    else:
        hero_path, supporting_paths = _generate_placeholder_images(
            brief, slug, images_dir
        )

    return {
        "hero_path": hero_path,
        "supporting_paths": supporting_paths,
    }


# -----------------------------
#  Provider Implementations
# -----------------------------

def _generate_openai_images(
    brief: str, slug: str, target_dir: Path
) -> Tuple[Path, List[Path]]:
    """
    Stub version.
    Replace with your real OpenAI image call.
    """

    print("⚠️ Using OPENAI stub image generator")

    # For now, fallback to placeholder (no crash)
    return _generate_placeholder_images(brief, slug, target_dir)


def _generate_gemini_images(
    brief: str, slug: str, target_dir: Path
) -> Tuple[Path, List[Path]]:
    """
    Stub version.
    Replace with your real Gemini call.
    """

    print("⚠️ Using GEMINI stub image generator")

    # For now, fallback to placeholder
    return _generate_placeholder_images(brief, slug, target_dir)


# -----------------------------
#  Placeholder Generator
# -----------------------------

def _generate_placeholder_images(
    brief: str,
    slug: str,
    target_dir: Path,
) -> Tuple[Path, List[Path]]:

    print("🟧 Using placeholder image generator")

    # Hero image
    hero_path = _create_placeholder_image(
        target_dir / f"{slug}_hero.png",
        f"{slug} HERO"
    )

    supporting_paths = []
    for i in range(1, 4):
        p = _create_placeholder_image(
            target_dir / f"{slug}_support_{i}.png",
            f"{slug} SUPPORT {i}"
        )
        supporting_paths.append(p)

    return hero_path, supporting_paths


# -----------------------------
#  Helper: Create Dummy Image
# -----------------------------

def _create_placeholder_image(path: Path, label: str) -> Path:
    """
    Creates a 1024×1024 colored box with centered text.
    Using PIL for simple placeholders.
    """

    img = Image.new("RGB", (1024, 1024), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)

    text = label.upper()
    text_w, text_h = draw.textbbox((0, 0), text)[2:]

    draw.text(
        ((1024 - text_w) / 2, (1024 - text_h) / 2),
        text,
        fill=(0, 0, 0)
    )

    img.save(path)
    return path
