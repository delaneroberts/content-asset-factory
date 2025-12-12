# ui/pages/04_Image_Library.py
from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import List, Tuple

import io
import json
import zipfile

import requests
import streamlit as st
from openai import OpenAI, OpenAIError
from huggingface_hub import InferenceClient
from PIL import Image

from caf_app.storage import load_campaign
from caf_app.models import Campaign  # for type hints / future use

# ---- Custom CSS for gallery improvements ----

#############
st.markdown("""
<style>
/* Add horizontal and vertical spacing between image cards */
.image-card {
    padding: 6px 12px 16px 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    border-radius: 10px;
    border: 1px solid #e5e7eb;  /* default subtle border */
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

/* Gold border for favorites */
.image-card.favorite {
    border: 3px solid #fbbf24;      /* gold-ish */
    box-shadow: 0 0 0 2px rgba(251, 191, 36, 0.35);
}

/* Container around the image */
.image-wrapper {
    position: relative;
    display: inline-block;
}

/* (Removed .favorite-badge – we don't need the floating star anymore) */
</style>
""", unsafe_allow_html=True)

##########
# ---------------------------------------------------------------------------
# Config / clients
# ---------------------------------------------------------------------------

client = OpenAI()

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
HF_NANOBANANA_MODEL_ID = os.getenv("HF_NANOBANANA_MODEL_ID")


# ---------------------------------------------------------------------------
# Path + metadata helpers
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
    JSON file that tracks pinned / favorite status and other metadata
    per image for this campaign.
    """
    base = _campaigns_root() / slug
    base.mkdir(parents=True, exist_ok=True)
    return base / "image_metadata.json"


def _load_image_metadata(slug: str) -> dict:
    """
    Load image metadata for a campaign, keyed by filename.
    Example:
        {
          "123456.png": {
              "pinned": true,
              "favorite": false,
              "engine": "openai",
              "prompt": "...",
              "kind": "generated",
              "created_at": "2025-12-10T22:15:02Z"
          },
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


def _update_image_metadata_entry(slug: str, filename: str, **fields) -> None:
    """
    Helper to merge fields into a single image's metadata entry
    without clobbering existing flags like pinned/favorite.
    """
    meta = _load_image_metadata(slug)
    info = meta.get(filename, {})
    info.update(fields)
    meta[filename] = info
    _save_image_metadata(slug, meta)


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
                    image=f,
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

    hf_client = InferenceClient(api_key=HF_API_KEY)

    images: List[bytes] = []
    for _ in range(n_images):
        img = hf_client.text_to_image(
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


# ---------------------------------------------------------------------------
# Storage helpers
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
    Save an uploaded image into the base images dir (not generated)
    and record basic metadata.
    """
    images_dir, _ = _campaign_dirs(slug)
    suffix = Path(uploaded_file.name).suffix or ".png"
    ts = int(time.time() * 1000)
    filename = f"uploaded_{ts}{suffix}"
    dest = images_dir / filename
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())

    _update_image_metadata_entry(
        slug,
        filename,
        kind="uploaded",
        original_filename=uploaded_file.name,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    return dest


def _build_images_zip(slug: str) -> bytes | None:
    """
    Package all images for a campaign into a ZIP file and return the bytes.
    """
    images = _list_all_images(slug)
    if not images:
        return None

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path in images:
            zf.write(img_path, arcname=img_path.name)

    buf.seek(0)
    return buf.getvalue()


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

    if campaign is None:
        st.warning(
            f"Could not load campaign for slug '{slug}'. "
            "It may have been deleted or the JSON file is missing."
        )
        return

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
        help=(
            "These images will appear in the gallery below. "
            "You can pin any of them and then use pinned images as bases for variants."
        ),
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
                    _update_image_metadata_entry(
                        slug,
                        path.name,
                        kind="generated",
                        engine=engine,
                        prompt=prompt.strip(),
                        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )

            st.success(f"Saved {len(saved_paths)} image(s) to this campaign.")
            st.rerun()
def _render_variant_generation_ui(slug: str) -> None:
    st.markdown("### 🪄 Generate Variants From a Base Image")

    with st.expander("Open variant generator", expanded=False):
        all_images = _list_all_images(slug)
        meta = _load_image_metadata(slug)

        # Use selected images (from the gallery checkboxes) as candidates
        selected_images = [p for p in all_images if meta.get(p.name, {}).get("selected")]

        base_image_path: Path | None = None

        if selected_images:
            st.write("Selected images (choose one as the base):")

            cols = st.columns(min(len(selected_images), 4))
            for idx, p in enumerate(selected_images):
                with cols[idx % len(cols)]:
                    st.image(str(p), caption=p.name, use_container_width=True)

            selected_name = st.radio(
                "Base image",
                options=[p.name for p in selected_images],
                index=0,
            )
            for p in selected_images:
                if p.name == selected_name:
                    base_image_path = p
                    break
        else:
            st.info(
                "To use a base image, first go to the image gallery below and "
                "check **Select** under the image you want to use. Then return here."
            )

        instructions = st.text_area(
            "How should we improve this image?",
            placeholder=(
                "e.g., Improve lighting, modernize the color palette, "
                "keep composition and main subject."
            ),
            height=100,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Engine**")
            st.write("OpenAI (Image Edit – preserves original)")
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
                    "No base image selected. Select at least one image in the gallery and pick it above."
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
                    _update_image_metadata_entry(
                        slug,
                        path.name,
                        kind="variant",
                        engine=engine,
                        instructions=instructions.strip(),
                        base_image=base_image_path.name,
                        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )

            st.success(f"Saved {len(saved_paths)} improved variant(s).")
            st.rerun()

def _render_export_section(slug: str) -> None:
    st.markdown("### 📦 Export Campaign Images")

    images = _list_all_images(slug)
    if not images:
        st.info("No images to export yet. Generate or upload images first.")
        return

    st.write(f"This campaign currently has **{len(images)}** image(s).")

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

    all_images = _list_all_images(slug)
    meta = _load_image_metadata(slug)

    if not all_images:
        st.info("No images yet. Generate or upload images first.")
        return

    # Filter favorites
    show_only_favorites = st.checkbox("Show only favorites", value=False)
    if show_only_favorites:
        images = [p for p in all_images if meta.get(p.name, {}).get("favorite")]
    else:
        images = all_images

    if not images:
        st.info("No images match this filter yet.")
        return

    # ---------- Preview + Info Panels ----------
    preview_key = f"{slug}_preview_image"
    info_key = f"{slug}_info_image"

    preview_path = st.session_state.get(preview_key)
    info_name = st.session_state.get(info_key)

    if preview_path:
        st.markdown("#### 🔍 Preview")
        st.image(preview_path, use_container_width=True)
        if st.button("Close preview"):
            st.session_state[preview_key] = None
            st.rerun()
        st.markdown("---")

    if info_name and info_name in meta:
        info = meta[info_name]
        st.markdown("#### ℹ️ Image info")
        st.write(f"**File:** `{info_name}`")

        if "kind" in info:
            st.write(f"**Kind:** {info['kind']}")
        if "engine" in info:
            st.write(f"**Engine:** {info['engine']}")

        if "prompt" in info:
            with st.expander("Prompt", expanded=False):
                st.write(info["prompt"])

        if "instructions" in info:
            with st.expander("Instructions", expanded=False):
                st.write(info["instructions"])

        if "base_image" in info:
            st.write(f"**Base image:** `{info['base_image']}`")

        if "original_filename" in info:
            st.write(f"**Original filename:** `{info['original_filename']}`")

        if "created_at" in info:
            st.write(f"**Created at:** {info['created_at']} (UTC)")

        if st.button("Close info"):
            st.session_state[info_key] = None
            st.rerun()

        st.markdown("---")

    # ---------- Sorting ----------
    # Do NOT sort by "selected" (keeps layout stable)
    def sort_key(p: Path):
        info = meta.get(p.name, {})
        favorite = bool(info.get("favorite"))
        return (0 if favorite else 1, -p.stat().st_mtime)

    images_sorted = sorted(images, key=sort_key)
    
    # ---------- CSS for centered image thumbnails ----------
    st.markdown("""
<style>
.caf-thumb-box {
    border-radius: 8px;
    border: 1px solid #ddd;
    overflow: hidden;
    width: 200px;
    height: 200px;
    margin: 0 auto 0.25rem auto;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #fafafa;
}
.caf-thumb-box img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center center;
}
.caf-link-row button {
    border: none;
    background: none;
    padding: 0 4px;
    margin: 0;
    font-size: 0.9rem;
}
.favorite-badge {
    position: absolute;
    top: 4px;
    right: 6px;
    background: rgba(255, 221, 0, 0.85);
    padding: 2px 6px;
    border-radius: 6px;
}
.image-card {
    padding-bottom: 10px;
}
.image-wrapper {
    position: relative;
}
</style>
""", unsafe_allow_html=True)
    
    selection_states = {}
    meta_changed = False

    # ---------- Main layout: gallery (left) + inspector (right) ----------
    gallery_col, inspector_col = st.columns([5, 2], gap="large")
#indentation conroversy
    with gallery_col:
        # ---------- 4-column responsive grid ----------
        cols = st.columns(4)

        for idx, img_path in enumerate(images_sorted):
            col = cols[idx % 4]
            with col:
                info = meta.get(img_path.name, {})
                favorite = bool(info.get("favorite"))
                selected = bool(info.get("selected"))

                badge = ""
                if favorite and selected:
                    badge = "⭐✔︎"
                elif favorite:
                    badge = "⭐"
                elif selected:
                    badge = "✔︎"

                caption = f"{badge} {img_path.name}" if badge else img_path.name

                st.markdown('<div class="image-card">', unsafe_allow_html=True)
                st.markdown('<div class="image-wrapper">', unsafe_allow_html=True)

                b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
                st.markdown(
                    f'<div class="caf-thumb-box"><img src="data:image/png;base64,{b64}"></div>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

                # ---------- Action Icons ----------
                lc1, lc2, lc3 = st.columns(3)

                with lc1:
                    if st.button("🔍", key=f"view_{slug}_{img_path.name}"):
                        st.session_state[preview_key] = str(img_path)
                        st.rerun()

                with lc2:
                    star_label = "⭐" if favorite else "☆"
                    if st.button(star_label, key=f"fav_{slug}_{img_path.name}"):
                        info["favorite"] = not favorite
                        meta[img_path.name] = info
                        meta_changed = True

                with lc3:
                    if st.button("ℹ️", key=f"info_{slug}_{img_path.name}"):
                        st.session_state["selected_image_name"] = img_path.name
                        st.rerun()


                # ---------- Select Checkbox ----------
                sel_key = f"sel_{slug}_{img_path.name}"
                sel_value = st.checkbox("Select", key=sel_key, value=selected)
                selection_states[img_path.name] = sel_value

                st.caption(caption)
                st.markdown('</div>', unsafe_allow_html=True)
                # END gallery_col
                # (everything above is inside with gallery_col)

#delane
    with inspector_col:

        st.subheader("Inspector")

        selected_name = st.session_state.get("selected_image_name")

        if not selected_name:
            st.caption("Click ℹ️ on an image to view/edit metadata.")
        else:
            info = meta.get(selected_name, {})

            # Find the selected image path by filename
            selected_path = next((p for p in images_sorted if p.name == selected_name), None)

            if selected_path and selected_path.exists():
                st.image(str(selected_path), use_container_width=True)

            st.markdown("### Provenance (read-only)")
            st.text_input("Filename", value=selected_name, disabled=True, key=f"ins_fn_{slug}_{selected_name}")
            st.text_input("Engine", value=str(info.get("engine", "")), disabled=True, key=f"ins_engine_{slug}_{selected_name}")
            st.text_area("Prompt", value=str(info.get("prompt", "")), disabled=True, height=120, key=f"ins_prompt_{slug}_{selected_name}")

            st.markdown("### Metadata (editable)")
            title = st.text_input("Title", value=str(info.get("title", "")), key=f"ins_title_{slug}_{selected_name}")

            type_options = ["hero", "lifestyle", "product_only", "background", "social", "supporting"]
            current_type = info.get("asset_type", "supporting")
            if current_type not in type_options:
                current_type = "supporting"

            asset_type = st.selectbox(
                "Type",
                type_options,
                index=type_options.index(current_type),
                key=f"ins_type_{slug}_{selected_name}",
            )

            channels = st.multiselect(
                "Channels",
                ["social", "web", "email", "print", "ads"],
                default=info.get("channels", []),
                key=f"ins_channels_{slug}_{selected_name}",
            )

            approval_options = ["draft", "approved", "rejected"]
            current_approval = info.get("approval_status", "draft")
            if current_approval not in approval_options:
                current_approval = "draft"

            approval_status = st.selectbox(
                "Approval status",
                approval_options,
                index=approval_options.index(current_approval),
                key=f"ins_approval_{slug}_{selected_name}",
            )

            rights_options = ["internal", "external", "paid_media", "unrestricted"]
            current_rights = info.get("usage_rights", "internal")
            if current_rights not in rights_options:
                current_rights = "internal"

            usage_rights = st.selectbox(
                "Usage rights",
                rights_options,
                index=rights_options.index(current_rights),
                key=f"ins_rights_{slug}_{selected_name}",
            )

            notes = st.text_area("Notes", value=str(info.get("notes", "")), height=120, key=f"ins_notes_{slug}_{selected_name}")

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Save", key=f"ins_save_{slug}_{selected_name}"):
                    info = meta.get(selected_name, {})
                    info["title"] = title
                    info["asset_type"] = asset_type
                    info["channels"] = channels
                    info["approval_status"] = approval_status
                    info["usage_rights"] = usage_rights
                    info["notes"] = notes
                    meta[selected_name] = info
                    meta_changed = True
                    st.success("Saved.")

            with c2:
                if st.button("Close", key=f"ins_close_{slug}_{selected_name}"):
                    st.session_state["selected_image_name"] = None
                    st.rerun()


    # ---------- Sync selection metadata ----------
    for filename, selected in selection_states.items():
        info = meta.get(filename, {})
        if selected and not info.get("selected"):
            info["selected"] = True
            meta_changed = True
        if not selected and info.get("selected"):
            info["selected"] = False
            meta_changed = True
        meta[filename] = info

    # ---------- Bulk Delete ----------
    if st.button("Delete selected images"):
        to_delete = [name for name, sel in selection_states.items() if sel]

        if not to_delete:
            st.info("No images selected.")
        else:
            for img_path in images_sorted:
                if img_path.name in to_delete:
                    try:
                        img_path.unlink()
                    except FileNotFoundError:
                        pass
                    meta.pop(img_path.name, None)

            _save_image_metadata(slug, meta)
            st.success(f"Deleted {len(to_delete)} image(s).")
            st.rerun()

    # Save metadata after favorites or selection changes
    if meta_changed:
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
    
    # Inspector state (right panel selection)
    if "selected_image_name" not in st.session_state:
        st.session_state["selected_image_name"] = None

    _render_campaign_header(slug)

    st.divider()
    _render_tools_panel(slug)

    st.divider()
    _render_gallery(slug)


if __name__ == "__main__":
    main()
