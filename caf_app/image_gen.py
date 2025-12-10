# caf_app/image_gen.py
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Tuple

import requests
from PIL import Image
from openai import OpenAI, OpenAIError

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

# OpenAI (gpt-image-1)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI()  # uses OPENAI_API_KEY from env

# Hugging Face NanoBanana
HF_API_KEY = os.getenv("HF_API_KEY")
HF_NANOBANANA_MODEL_ID = os.getenv(
    "HF_NANOBANANA_MODEL_ID", "REPLACE_WITH_NANOBANANA_MODEL_ID"
)

# Stability / SDXL
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
STABILITY_ENGINE_ID = os.getenv(
    "STABILITY_ENGINE_ID",
    "stable-diffusion-xl-1024-v1-0",  # common SDXL engine id; override in .env if needed
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_size(size: str) -> Tuple[int, int]:
    """
    Convert a size string like '1024x1024' into (width, height).
    """
    try:
        w_str, h_str = size.lower().split("x")
        return int(w_str), int(h_str)
    except Exception:
        # Fallback to 1024x1024 if something weird comes in
        return 1024, 1024


# ---------------------------------------------------------------------------
# OpenAI generator (gpt-image-1)
# ---------------------------------------------------------------------------

def _generate_openai_image(prompt: str, size: str) -> bytes:
    """
    Generate an image using OpenAI gpt-image-1.

    NOTE: New API does NOT accept 'response_format', so we omit it.
    By default it returns base64 JSON (b64_json). If for some reason it
    returns URLs instead, we fall back to fetching the URL.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    try:
        result = openai_client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=size,
            n=1,
        )
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI image generation failed: {e}") from e

    # Preferred: base64 field
    try:
        b64 = result.data[0].b64_json
        return base64.b64decode(b64)
    except Exception:
        # Fallback: URL-based response
        try:
            url = result.data[0].url
        except Exception:
            raise RuntimeError("OpenAI image response missing 'b64_json' and 'url'.")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content


# ---------------------------------------------------------------------------
# Hugging Face NanoBanana generator (via router.huggingface.co)
# ---------------------------------------------------------------------------

def _generate_nanobanana_image(prompt: str, size: str) -> bytes:
    """
    Generate an image using a NanoBanana model on Hugging Face Inference.

    Uses the new router endpoint:
      https://router.huggingface.co/hf-inference/models/{model_id}
    """
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY is not set in the environment.")

    if HF_NANOBANANA_MODEL_ID in ("", "REPLACE_WITH_NANOBANANA_MODEL_ID", None):
        raise RuntimeError(
            "HF_NANOBANANA_MODEL_ID is not configured. "
            "Set it to your NanoBanana model ID from Hugging Face."
        )

    width, height = _parse_size(size)

    # NEW REQUIRED ENDPOINT
    url = f"https://router.huggingface.co/hf-inference/models/{HF_NANOBANANA_MODEL_ID}"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Accept": "image/png",
    }

    # Most image models on HF inference accept this minimal JSON structure.
    payload = {
        "inputs": prompt,
        "options": {
            "wait_for_model": True,
        },
        # Width/height are accepted by many SDXL fine-tunes; others may ignore.
        "width": width,
        "height": height,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)

    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"NanoBanana generation failed: {resp.status_code} {detail}")

    # We requested Accept: image/png, so resp.content is raw PNG bytes
    return resp.content


# ---------------------------------------------------------------------------
# Stability / SDXL generator
# ---------------------------------------------------------------------------

def _generate_stability_image(prompt: str, size: str) -> bytes:
    """
    Generate an image using Stability's text-to-image SDXL endpoint.
    """
    if not STABILITY_API_KEY:
        raise RuntimeError("STABILITY_API_KEY is not set in the environment.")

    width, height = _parse_size(size)

    url = f"https://api.stability.ai/v1/generation/{STABILITY_ENGINE_ID}/text-to-image"

    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7,
        "clip_guidance_preset": "NONE",
        "height": height,
        "width": width,
        "samples": 1,
        "steps": 30,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)

    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"Stability SDXL generation failed: {resp.status_code} {detail}")

    data = resp.json()
    artifacts = data.get("artifacts", [])
    if not artifacts:
        raise RuntimeError("Stability SDXL response contained no artifacts.")

    b64 = artifacts[0].get("base64")
    if not b64:
        raise RuntimeError("Stability SDXL artifact missing 'base64' field.")

    return base64.b64decode(b64)


# ---------------------------------------------------------------------------
# Public multi-engine adapter
# ---------------------------------------------------------------------------

def generate_image(
    prompt: str,
    engine: str = "auto",
    size: str = "1024x1024",
) -> tuple[bytes, str]:
    """
    Unified image generator used by CAF.

    Returns (image_bytes, engine_used).

    engine:
      - "openai"       → gpt-image-1
      - "nanobanana"   → HF NanoBanana
      - "stability"/"sdxl" → Stability SDXL
      - "auto"         → try OpenAI → NanoBanana → Stability
    """
    engine = (engine or "auto").lower().strip()

    # Explicit engine selection
    if engine == "openai":
        return _generate_openai_image(prompt, size), "openai"

    if engine == "nanobanana":
        return _generate_nanobanana_image(prompt, size), "nanobanana"

    if engine in ("stability", "sdxl"):
        return _generate_stability_image(prompt, size), "stability"

    # "auto" mode: cascade through providers
    errors: list[str] = []

    # 1) Try OpenAI
    try:
        img = _generate_openai_image(prompt, size)
        return img, "openai"
    except Exception as e:
        errors.append(f"OpenAI: {e}")

    # 2) Try NanoBanana
    try:
        img = _generate_nanobanana_image(prompt, size)
        return img, "nanobanana"
    except Exception as e:
        errors.append(f"NanoBanana: {e}")

    # 3) Try Stability / SDXL
    try:
        img = _generate_stability_image(prompt, size)
        return img, "stability"
    except Exception as e:
        errors.append(f"Stability: {e}")

    # If everything failed, surface a concise summary
    raise RuntimeError(
        "All image engines failed in auto mode:\n" + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# Utility: save bytes to PNG on disk
# ---------------------------------------------------------------------------

def save_png(image_bytes: bytes, out_path: Path) -> None:
    """
    Save image bytes as a PNG file at out_path.

    Handles both already-encoded PNGs and raw image bytes that Pillow can open.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGBA")
            img.save(out_path, format="PNG")
    except Exception:
        # As a fallback, just write raw bytes (assume PNG)
        with out_path.open("wb") as f:
            f.write(image_bytes)


# ---------------------------------------------------------------------------
# Utility: Adobe Firefly handoff URL
# ---------------------------------------------------------------------------

def make_firefly_edit_url(image_path: Path) -> str:
    """
    Construct a Firefly URL for editing.

    Today this just opens Firefly's main page with a hint in the query;
    Firefly does not support a direct local-file deep link without upload.
    """
    return f"https://firefly.adobe.com/?ref=content-asset-factory&image={image_path.name}"
