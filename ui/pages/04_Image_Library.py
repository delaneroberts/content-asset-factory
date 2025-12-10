# ui/pages/04_Image_Library.py
from __future__ import annotations
import base64
import os
import time
from pathlib import Path
from typing import List, Tuple
import shutil
import requests
import streamlit as st
from openai import OpenAI, OpenAIError
import io
from huggingface_hub import InferenceClient
from caf_app.storage import load_campaign
from caf_app.models import Campaign  # imported for type hints / future use
import io
import zipfile
import json
from PIL import Image

# ---------------------------------------------------------------------------
# Config / clients
# ---------------------------------------------------------------------------

client = OpenAI()

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
HF_NANOBANANA_MODEL_ID = os.getenv("HF_NANOBANANA_MODEL_ID")

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """
    Return the project root directory (one level above /ui).
    This file lives at <root>/ui/pages/04_Image_Library.py
    """
    return Path(__file__).resolve().parents[2]


def _campaigns_root() -> Path:
    root = _project_root()
    campaigns = root / "campaigns"
    campaigns.mkdir(parents=True, exist_ok=True)
    return campaigns

def _metadata_path(slug: str) -> Path:
    """
    JSON file that tracks pinned / favorite status per image for this campaign.
    """
    base = _campaigns_root() / slug
    base.mkdir(parents=True, exist_ok=True)
    return base / "image_metadata.json"


def _load_image_metadata(slug: str) -> dict:
    """
    Load image metadata for a campaign, keyed by filename.
    Example:
        {
          "123456.png": {"pinned": true, "favorite": false},
          ...
        }
    """
    path = _metadata_path(slug)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        # If anything goes wrong, fall back to empty metadata rather than dying.
        return {}


def _save_image_metadata(slug: str, metadata: dict) -> None:
    """
    Persist image metadata to disk.
    """
    path = _metadata_path(slug)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _get_current_slug() -> str | None:
    """
    The Dashboard / other pages should set:
        st.session_state["current_campaign_slug"]
    """
    return st.session_state.get("current_campaign_slug")


def _campaign_dirs(slug: str) -> Tuple[Path, Path]:
    """
    Returns (images_dir, generated_dir) for a campaign.
    """
    base = _campaigns_root() / slug
    images_dir = base / "images"
    generated_dir = images_dir / "generated"
    images_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, generated_dir


def _list_all_images(slug: str) -> List[Path]:
    """
    Return all images (uploaded + generated) for a campaign as Paths,
    sorted newest-first by modification time.
    """
    images_dir, generated_dir = _campaign_dirs(slug)

    paths: List[Path] = []
    for directory in (images_dir, generated_dir):
        if directory.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                paths.extend(directory.glob(ext))

    # Deduplicate and sort
    unique = list({p.resolve(): p for p in paths}.values())
    unique.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return unique


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------

def _stability_variants_from_base(
    base_image_path: Path,
    instructions: str,
    n_images: int,
) -> List[bytes]:
    """
    Use Stability's edit/inpaint endpoint to generate variants starting from
    a base image. This tends to preserve more of the original structure than
    pure text-to-image.
    """
    if not STABILITY_API_KEY:
        raise RuntimeError("STABILITY_API_KEY is not set in the environment.")

    url = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/*",
    }

    # Full-white mask: allow edits anywhere, but diffusion still starts from the base.
    base = Image.open(base_image_path).convert("RGBA")
    mask = Image.new("L", base.size, color=255)
    mask_buf = io.BytesIO()
    mask.save(mask_buf, format="PNG")
    mask_bytes = mask_buf.getvalue()

    images: List[bytes] = []

    for _ in range(n_images):
        with open(base_image_path, "rb") as f:
            files = {
                "image": ("image.png", f, "image/png"),
                "mask": ("mask.png", mask_bytes, "image/png"),
            }
            data = {
                "prompt": instructions,
                "output_format": "png",
            }

            resp = requests.post(
                url,
                headers=headers,
                files=files,
                data=data,
                timeout=120,
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Stability edit failed: {resp.status_code} {resp.text[:500]}"
            )

        images.append(resp.content)

    return images

def _openai_variants_from_base(
    base_image_path: Path,
    instructions: str,
    n_images: int,
) -> List[bytes]:
    """
    Use OpenAI's image edit API (gpt-image-1) to generate variants that
    preserve the original image as much as possible while applying the
    requested improvements.
    """
    images: List[bytes] = []

    for _ in range(n_images):
        with open(base_image_path, "rb") as f:
            try:
                result = client.images.edit(
                    model="gpt-image-1",
                    image=f,              # 👈 single image file object
                    prompt=instructions,
                    n=1,
                    size="1024x1024",
                )
            except OpenAIError as e:
                raise RuntimeError(f"OpenAI variant edit failed: {e}") from e

        # gpt-image-1 always returns base64
        b64 = result.data[0].b64_json
        images.append(base64.b64decode(b64))

    return images


def _generate_openai_images(prompt: str, n_images: int) -> List[bytes]:
    if n_images < 1:
        return []

    try:
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=n_images,
            size="1024x1024",
            # NOTE: no response_format param to avoid the 400 unknown_parameter error
        )
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI image generation failed: {e}") from e

    images: List[bytes] = []
    for item in response.data:
        b64 = item.b64_json
        images.append(base64.b64decode(b64))
    return images

def _generate_stability_images(prompt: str, n_images: int) -> List[bytes]:
    if not STABILITY_API_KEY:
        raise RuntimeError("STABILITY_API_KEY is not set in the environment.")

    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        # Stability requires an explicit Accept header: image/* or application/json
        "Accept": "image/*",
    }

    images: List[bytes] = []
    for _ in range(n_images):
        files = {
            "prompt": (None, prompt),
            "output_format": (None, "png"),
        }
        resp = requests.post(url, headers=headers, files=files, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Stability API error {resp.status_code}: {resp.text[:500]}"
            )
        images.append(resp.content)
    return images



def _generate_nanobanana_images(prompt: str, n_images: int) -> List[bytes]:
    """
    NanoBanana / FLUX via Hugging Face InferenceClient.

    Uses HF_API_KEY as the token and HF_NANOBANANA_MODEL_ID as the model id,
    e.g. "black-forest-labs/FLUX.1-dev".
    """
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY is not set in the environment.")

    if not HF_NANOBANANA_MODEL_ID or "REPLACE_WITH" in HF_NANOBANANA_MODEL_ID:
        raise RuntimeError(
            "HF_NANOBANANA_MODEL_ID is not configured. "
            "Set it to your actual model id on Hugging Face, e.g. 'black-forest-labs/FLUX.1-dev'."
        )

    client = InferenceClient(api_key=HF_API_KEY)

    images: List[bytes] = []
    for _ in range(n_images):
        # This returns a PIL.Image
        img = client.text_to_image(
            prompt=prompt,
            model=HF_NANOBANANA_MODEL_ID,
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images.append(buf.getvalue())

    return images


def _generate_images_for_engine(
    engine: str, prompt: str, n_images: int
) -> List[bytes]:
    engine = engine.lower()
    if engine == "openai":
        return _generate_openai_images(prompt, n_images)
    elif engine == "stability":
        return _generate_stability_images(prompt, n_images)
    elif engine == "nanobanana":
        return _generate_nanobanana_images(prompt, n_images)
    else:
        raise ValueError(f"Unknown engine: {engine}")

def _generate_variants_for_engine(
    engine: str,
    base_image_path: Path,
    instructions: str,
    n_images: int,
) -> List[bytes]:
    """
    For now, only OpenAI is used for true image edits.
    """
    engine = engine.lower()
    if engine != "openai":
        raise RuntimeError("Variant-from-base currently supports only OpenAI image edits.")
    return _openai_variants_from_base(base_image_path, instructions, n_images)


    # Fallback for stability / nanobanana: treat as “inspired by base”
    base_hint = (
        "Create a new image inspired by a reference image, preserving the main subject and "
        f"overall composition, but applying the following changes: {instructions.strip()}"
    )
    return _generate_images_for_engine(engine, base_hint, n_images)


# ---------------------------------------------------------------------------
# Storage helpers for generated / uploaded images
# ---------------------------------------------------------------------------


def _save_image_bytes(slug: str, img_bytes: bytes, generated: bool = True) -> Path:
    """
    Save image bytes into the campaign folder and return the Path.
    If generated=True, save under images/generated; else under images/.
    """
    images_dir, generated_dir = _campaign_dirs(slug)
    target_dir = generated_dir if generated else images_dir

    ts = int(time.time() * 1000)
    filename = f"{ts}.png"
    path = target_dir / filename

    with open(path, "wb") as f:
        f.write(img_bytes)

    return path


def _save_uploaded_image(slug: str, uploaded_file) -> Path:
    """
    Save an uploaded image into the base images dir (not generated).
    """
    images_dir, _ = _campaign_dirs(slug)
    suffix = Path(uploaded_file.name).suffix or ".png"
    ts = int(time.time() * 1000)
    filename = f"uploaded_{ts}{suffix}"
    dest = images_dir / filename
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------
def _render_tools_panel(slug: str) -> None:
    """
    Compact wrapper that puts all the 'tool' sections behind a single expander
    with tabs, so the top of the page stays small.
    """
    with st.expander(
        "Tools: upload / generate / variants / export",
        expanded=False,
    ):
        tabs = st.tabs(
            [
                "Upload external images",
                "Generate from prompt",
                "Variants from base",
                "Export images",
            ]
        )

        with tabs[0]:
            _render_upload_section(slug)

        with tabs[1]:
            _render_prompt_generation_ui(slug)

        with tabs[2]:
            _render_variant_generation_ui(slug)

        with tabs[3]:
            _render_export_section(slug)


def _render_campaign_header(slug: str) -> None:
    campaign = load_campaign(slug)

    # If campaign couldn't be loaded, don't crash the page.
    if campaign is None:
        st.warning(f"Could not load campaign for slug '{slug}'. "
                   "It may have been deleted or the JSON file is missing.")
        return

    # Try to find a reasonable name/title field
    name = None
    for attr in ("name", "title", "campaign_name"):
        if hasattr(campaign, attr):
            name = getattr(campaign, attr)
            break

    if not name and isinstance(getattr(campaign, "model_dump", None), callable):
        data = campaign.model_dump()
        for key in ("name", "title", "campaign_name"):
            if key in data and data[key]:
                name = data[key]
                break

    if not name:
        name = slug

    st.subheader(f"Campaign: {name}")

    # --- brief / description (same defensive pattern) ---
    brief = None
    for attr in ("campaign_brief", "brief", "description", "summary"):
        if hasattr(campaign, attr):
            brief = getattr(campaign, attr)
            break

    if brief is None and isinstance(getattr(campaign, "model_dump", None), callable):
        data = campaign.model_dump()
        for key in ("campaign_brief", "brief", "description", "summary"):
            if key in data and data[key]:
                brief = data[key]
                break

    if brief:
        with st.expander("View campaign brief", expanded=False):
            st.write(brief)


def _render_upload_section(slug: str) -> None:
    st.markdown("### ⬆️ Upload External Images")

    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="These images will appear in the gallery below. You can pin any of them and then use pinned images as bases for variants.",
    )

    if uploaded_files:
        if st.button("Save uploaded images"):
            count = 0
            for f in uploaded_files:
                _save_uploaded_image(slug, f)
                count += 1
            st.success(f"Saved {count} image(s). They now appear in the gallery.")
            st.rerun()


def _render_prompt_generation_ui(slug: str) -> None:
    st.markdown("### 🎨 Generate Images From Prompt")

    with st.expander("Open generator", expanded=False):
        prompt = st.text_area(
            "Prompt",
            placeholder="Describe the image you want to generate…",
            height=120,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            engine = st.selectbox(
                "Engine",
                ["stability", "openai", "nanobanana"],
                index=0,
                help="Which image engine to use.",
            )
        with col2:
            n_images = st.slider(
                "Number of images",
                min_value=1,
                max_value=6,
                value=2,
            )
        with col3:
            size = st.selectbox(
                "Size (ignored by some engines)",
                ["1024x1024", "768x768"],
                index=0,
            )

        if st.button("Generate images", type="primary"):
            if not prompt.strip():
                st.warning("Please enter a prompt.")
                return

            with st.spinner(f"Generating {n_images} image(s) with {engine}…"):
                try:
                    img_bytes_list = _generate_images_for_engine(engine, prompt, n_images)
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))
                    return

                saved_paths = []
                for img_bytes in img_bytes_list:
                    path = _save_image_bytes(slug, img_bytes, generated=True)
                    saved_paths.append(path)

            st.success(f"Saved {len(saved_paths)} image(s) to this campaign.")
            st.rerun()

def _render_variant_generation_ui(slug: str) -> None:
    st.markdown("### 🪄 Generate Variants From a Base Image")

    with st.expander("Open variant generator", expanded=False):
        # Use pinned images as candidate bases
        all_images = _list_all_images(slug)
        meta = _load_image_metadata(slug)
        pinned_images = [p for p in all_images if meta.get(p.name, {}).get("pinned")]

        base_image_path: Path | None = None

        if pinned_images:
            st.write("Pinned images (choose one as the base):")

            # Show thumbnails of pinned images
            cols = st.columns(min(len(pinned_images), 4))
            for idx, p in enumerate(pinned_images):
                with cols[idx % len(cols)]:
                    st.image(str(p), caption=p.name, use_container_width=True)

            selected_name = st.radio(
                "Base image",
                options=[p.name for p in pinned_images],
                index=0,
            )
            for p in pinned_images:
                if p.name == selected_name:
                    base_image_path = p
                    break

        else:
            st.info(
                "To use a base image, first upload images (above) or generate them, "
                "then pin one in the gallery below. Pinned images will appear here as choices."
            )

        instructions = st.text_area(
            "How should we improve this image?",
            placeholder="e.g., Improve lighting, modernize the color palette, keep composition and main subject.",
            height=100,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Engine**")
            st.write("OpenAI (Image Edit – preserves original)")  # label only
            engine = "openai"

        with col2:
            n_variants = st.slider(
                "Number of variants",
                min_value=1,
                max_value=6,
                value=2,
            )

        if st.button("Generate improved variants", type="primary"):
            if not base_image_path:
                st.warning(
                    "No base image selected. Pin an image in the gallery and then choose it here."
                )
                return
            if not instructions.strip():
                st.warning("Please describe how to improve the image.")
                return

            with st.spinner(f"Generating {n_variants} improved image(s) with OpenAI…"):
                try:
                    img_bytes_list = _generate_variants_for_engine(
                        engine,
                        base_image_path,
                        instructions.strip(),
                        n_variants,
                    )
                except Exception as e:
                    st.error(str(e))
                    return

                saved_paths = []
                for img_bytes in img_bytes_list:
                    path = _save_image_bytes(slug, img_bytes, generated=True)
                    saved_paths.append(path)

            st.success(f"Saved {len(saved_paths)} improved variant(s).")
            st.rerun()



def _render_export_section(slug: str) -> None:
    st.markdown("### 📦 Export Campaign Images")

    images = _list_all_images(slug)
    if not images:
        st.info("No images to export yet. Generate or upload images first.")
        return

    st.write(f"This campaign currently has **{len(images)}** image(s).")

    # Build the zip lazily when user clicks
    if st.button("Prepare ZIP file"):
        with st.spinner("Building ZIP of campaign images…"):
            zip_bytes = _build_images_zip(slug)

        if not zip_bytes:
            st.warning("No images were found to export.")
            return

        st.success("ZIP file ready. Click below to download:")
        st.download_button(
            label="⬇️ Download campaign_images.zip",
            data=zip_bytes,
            file_name=f"{slug}_images.zip",
            mime="application/zip",
        )

def _render_gallery(slug: str) -> None:
    st.markdown("### 🖼️ Campaign Image Library")

    images = _list_all_images(slug)
    if not images:
        st.info("No images yet. Generate or upload images first.")
        return

    # Load metadata (pinned / favorite)
    meta = _load_image_metadata(slug)

    # ---------- Fullscreen preview ----------
    preview_key = f"{slug}_fullscreen_image"
    preview_path = st.session_state.get(preview_key)

    if preview_path:
        st.markdown("#### 🔍 Preview")
        st.image(preview_path, use_container_width=True)
        if st.button("❌ Close preview"):
            st.session_state[preview_key] = None
            st.rerun()

        st.markdown("---")

    # ---------- Sort images: pinned, favorites, newest ----------
    def sort_key(p: Path):
        info = meta.get(p.name, {})
        pinned = bool(info.get("pinned", False))
        favorite = bool(info.get("favorite", False))
        return (
            0 if pinned else 1,
            0 if favorite else 1,
            -p.stat().st_mtime,
        )

    images_sorted = sorted(images, key=sort_key)

    changed = False

    # Wider cards so buttons don't wrap
    cols = st.columns(2)

    for idx, img_path in enumerate(images_sorted):
        col = cols[idx % len(cols)]
        with col:
            info = meta.get(img_path.name, {})
            pinned = bool(info.get("pinned", False))
            favorite = bool(info.get("favorite", False))

            # Caption badges
            badge = ""
            if pinned and favorite:
                badge = "📌⭐"
            elif pinned:
                badge = "📌"
            elif favorite:
                badge = "⭐"

            caption = img_path.name
            if badge:
                caption = f"{badge} {caption}"

            st.image(str(img_path), caption=caption, use_container_width=True)

            # Row 1: View + Delete
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                if st.button("Preview", key=f"view_{slug}_{idx}"):
                    st.session_state[preview_key] = str(img_path)
                    st.rerun()

            with row1_col2:
                if st.button("🗑️", help="Delete image", key=f"del_{slug}_{idx}"):
                    try:
                        img_path.unlink()
                    except FileNotFoundError:
                        pass
                    meta.pop(img_path.name, None)
                    changed = True

            # Row 2: Pin + Favorite
            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                pin_label = "Pin" if not pinned else "Unpin"
                if st.button(pin_label, help="Pin / unpin image", key=f"pin_{slug}_{idx}"):
                    info = meta.get(img_path.name, {})
                    info["pinned"] = not pinned
                    meta[img_path.name] = info
                    changed = True

            with row2_col2:
                fav_label = "⭐" if not favorite else "⭐ Unfav"
                if st.button(fav_label, help="Mark / unmark favorite", key=f"fav_{slug}_{idx}"):
                    info = meta.get(img_path.name, {})
                    info["favorite"] = not favorite
                    meta[img_path.name] = info
                    changed = True

    if changed:
        _save_image_metadata(slug, meta)
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Image Library")

    slug = _get_current_slug()
    if not slug:
        st.warning("No campaign selected. Please choose a campaign on the Dashboard first.")
        return

    _render_campaign_header(slug)

    # Compact tools bar (one line) at the top
    st.divider()
    _render_tools_panel(slug)

    # Library visible without scrolling
    st.divider()
    _render_gallery(slug)


if __name__ == "__main__":
    main()
