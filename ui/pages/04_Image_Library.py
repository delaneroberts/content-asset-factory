# ui/pages/04_Image_Library.py
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from PIL import Image
from openai import OpenAI, OpenAIError

from caf_app.storage import campaigns_dir, load_campaign
from caf_app.image_gen import generate_image, save_png, make_firefly_edit_url

# OpenAI client (uses OPENAI_API_KEY from .env)
client = OpenAI()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _get_current_slug() -> str | None:
    return st.session_state.get("current_campaign_slug")


def _campaign_dir(slug: str) -> Path:
    return campaigns_dir() / slug


def _images_base_dir(slug: str) -> Path:
    return _campaign_dir(slug) / "images"


def _generated_dir(slug: str) -> Path:
    return _images_base_dir(slug) / "generated"


def _uploads_dir(slug: str) -> Path:
    return _images_base_dir(slug) / "uploads"


def _ensure_dirs(slug: str) -> None:
    for p in (_generated_dir(slug), _uploads_dir(slug)):
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pin / favorite metadata
# ---------------------------------------------------------------------------

def _meta_path(slug: str) -> Path:
    return _images_base_dir(slug) / "images_meta.json"


def _load_meta(slug: str) -> Dict[str, Dict[str, Any]]:
    mp = _meta_path(slug)
    if not mp.exists():
        return {}
    try:
        with mp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_meta(slug: str, meta: Dict[str, Dict[str, Any]]) -> None:
    mp = _meta_path(slug)
    mp.parent.mkdir(parents=True, exist_ok=True)
    with mp.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# List images + meta
# ---------------------------------------------------------------------------

def _list_images_with_meta(slug: str) -> List[Dict[str, Any]]:
    meta = _load_meta(slug)
    items: List[Dict[str, Any]] = []

    sections = [
        ("generated", _generated_dir),
        ("uploads", _uploads_dir),
    ]
    exts = ("*.png", "*.jpg", "*.jpeg")

    for section_name, dir_fn in sections:
        d = dir_fn(slug)
        if not d.exists():
            continue
        for pattern in exts:
            for p in sorted(d.glob(pattern)):
                filename = p.name
                entry_meta = meta.get(filename, {})
                items.append(
                    {
                        "path": p,
                        "section": section_name,
                        "pinned": bool(entry_meta.get("pinned", False)),
                        "favorite": bool(entry_meta.get("favorite", False)),
                    }
                )

    # Sort: pinned → favorite → filename
    items.sort(
        key=lambda x: (
            0 if x["pinned"] else 1,
            0 if x["favorite"] else 1,
            x["path"].name.lower(),
        )
    )
    return items


# ---------------------------------------------------------------------------
# OpenAI exact edit helper (gpt-image-1) — matches official Python example
# ---------------------------------------------------------------------------

def _edit_image_with_openai(
    template_path: Path,
    prompt: str,
    size: str = "1024x1024",
) -> bytes:
    """
    True in-place edit using OpenAI's image edit API.

    This is what gives you: "same image, but now with a top hat / balloon / worker".
    """
    try:
        # Important: pass image as a LIST of file handles, like the docs
        with template_path.open("rb") as f:
            result = client.images.edit(
                model="gpt-image-1",
                image=[f],
                prompt=prompt,
                size=size,
                n=1,
            )

        b64 = result.data[0].b64_json
        return base64.b64decode(b64)

    except OpenAIError as e:
        raise RuntimeError(f"OpenAI image edit failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error in image edit: {e}") from e


# ---------------------------------------------------------------------------
# Main Streamlit page
# ---------------------------------------------------------------------------

def main():
    st.title("Image Library")

    slug = _get_current_slug()
    if not slug:
        st.info("Select or create a campaign in the Dashboard first.")
        return

    campaign = load_campaign(slug)
    campaign_name = getattr(campaign, "name", slug)
    st.caption(f"Active campaign: **{campaign_name}**  (slug: `{slug}`)")

    _ensure_dirs(slug)

    # ----------------------------------------------------------------------
    # Load images + metadata
    # ----------------------------------------------------------------------
    images_with_meta = _list_images_with_meta(slug)
    meta = _load_meta(slug)
    pinned_images = [img for img in images_with_meta if img["pinned"]]

    # ----------------------------------------------------------------------
    # Sidebar: pinned sidecar (multiple pins supported)
    # ----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Pinned images")

        if pinned_images:
            primary = pinned_images[0]
            st.image(str(primary["path"]), use_container_width=True)
            st.caption(f"Primary pinned: {primary['path'].name}")

            if len(pinned_images) > 1:
                st.markdown("Other pinned images:")
                for extra in pinned_images[1:4]:
                    st.image(str(extra["path"]), use_container_width=True)
                    st.caption(extra["path"].name)
        else:
            st.caption("Pin one or more images below to feature them here.")

    # ----------------------------------------------------------------------
    # Generate / Edit images
    # ----------------------------------------------------------------------
    with st.expander("Generate new images", expanded=False):
        default_prompt = (
            getattr(campaign, "campaign_brief", "")
            or getattr(campaign, "brief", "")
            or ""
        )

        base_prompt = st.text_area(
            "Prompt",
            value=default_prompt,
            help=(
                "For edits, describe ONLY the change, e.g. "
                "'put a construction worker standing in front of the bulldozer, "
                "keep everything else the same.'"
            ),
            height=150,
        )

        engine_choice = st.selectbox(
            "Image Engine",
            options=["auto", "openai", "nanobanana", "stability"],
            index=0,
            help=(
                "auto: try OpenAI first, then fall back to NanoBanana, then Stability.\n"
                "openai: OpenAI only (supports exact edits when using the pinned image mode).\n"
                "nanobanana: NanoBanana (Hugging Face) only.\n"
                "stability: Stability SDXL only (great for photorealistic variants)."
            ),
        )

        num_images = st.slider(
            "Number of images",
            min_value=1,
            max_value=4,
            value=1,
            help="Generate up to 4 images in one batch.",
        )

        use_template = False
        exact_edit_mode = False

        if pinned_images:
            use_template = st.checkbox(
                "Use pinned images as style template / edit source",
                value=False,
                help=(
                    "When enabled:\n"
                    "• If engine is 'openai' or 'auto', CAF will edit the PRIMARY pinned image "
                    "in-place using GPT Image (same image + your change).\n"
                    "• If engine is 'nanobanana', CAF will generate new images in a similar style."
                ),
            )
            if use_template:
                st.info(
                    "Pinned template active.\n\n"
                    "• With OpenAI: true in-place edit of the primary pinned image.\n"
                    "• With NanoBanana: new images in a similar style (no exact edits)."
                )
                if engine_choice in ("auto", "openai"):
                    exact_edit_mode = True

        # Effective prompt for non-edit style-mode
        if use_template and pinned_images and not exact_edit_mode:
            names = ", ".join(img["path"].name for img in pinned_images[:4])
            effective_prompt = (
                "Create new campaign images that closely match the overall style, "
                "composition, color palette, and visual language of the pinned "
                f"images ({names}). The new images should feel like they belong in "
                "the same campaign system. Then apply the following instructions:\n\n"
                f"{base_prompt}"
            )
        else:
            effective_prompt = base_prompt

        col_gen, col_status = st.columns([1, 2])
        gen_clicked = col_gen.button("Generate", type="primary")

        st.write("DEBUG → gen_clicked:", gen_clicked, "| num_images:", num_images)

        if gen_clicked:
            if not base_prompt.strip():
                col_status.error("Please enter a prompt first.")
            else:
                try:
                    with col_status:
                        with st.spinner(
                            f"Generating {num_images} image(s) using '{engine_choice}'..."
                        ):
                            successes = 0
                            errors: List[str] = []

                            for i in range(num_images):
                                try:
                                    if exact_edit_mode and pinned_images:
                                        # TRUE EDIT: primary pinned image, gpt-image-1 edit
                                        edit_prompt = (
                                            "Edit this image while keeping everything else "
                                            "as close as possible to the original. Apply ONLY "
                                            "the following change(s):\n\n"
                                            f"{base_prompt}"
                                        )
                                        template_path: Path = pinned_images[0]["path"]
                                        image_bytes = _edit_image_with_openai(
                                            template_path=template_path,
                                            prompt=edit_prompt,
                                            size="1024x1024",
                                        )
                                    else:
                                        # Normal multi-engine generation (OpenAI / NanoBanana)
                                        image_bytes, _engine_used = generate_image(
                                            prompt=effective_prompt,
                                            engine=engine_choice,
                                            size="1024x1024",
                                        )

                                    ts = int(time.time() * 1000)
                                    filename = f"{slug}_{ts}_{i}.png"
                                    out_path = _generated_dir(slug) / filename
                                    save_png(image_bytes, out_path)
                                    successes += 1
                                except Exception as e:
                                    errors.append(str(e))

                            if successes:
                                if exact_edit_mode and pinned_images:
                                    msg_extra = " (exact edit of primary pinned image)"
                                elif use_template and pinned_images:
                                    msg_extra = " (style-guided by pinned images)"
                                else:
                                    msg_extra = ""
                                st.success(
                                    f"Generated {successes} image(s){msg_extra}. "
                                    "Scroll down to the Library to see them."
                                )
                            if errors:
                                st.error("Some images failed:\n\n" + "\n".join(errors))
                except Exception as e:
                    col_status.error(f"Unexpected error in generate handler: {e}")
                    st.exception(e)

    # ----------------------------------------------------------------------
    # Upload existing images
    # ----------------------------------------------------------------------
    with st.expander("Upload existing images", expanded=False):
        upload_files = st.file_uploader(
            "Upload PNG or JPG files",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        if upload_files and st.button("Save uploads"):
            status_box = st.empty()
            saved_count = 0

            for f in upload_files:
                try:
                    img = Image.open(f).convert("RGBA")
                    ts = int(time.time() * 1000)
                    filename = f"upload_{ts}_{f.name}.png"
                    out_path = _uploads_dir(slug) / filename
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    img.save(out_path, format="PNG")
                    saved_count += 1
                except Exception as e:
                    status_box.error(f"Failed to save {f.name}: {e}")

            if saved_count:
                status_box.success(
                    f"Saved {saved_count} uploaded image(s). "
                    "Scroll down to see them in the Library."
                )

    # ----------------------------------------------------------------------
    # Library / gallery with pin/favorite/delete/Firefly
    # ----------------------------------------------------------------------
    st.subheader("Library")

    images_with_meta = _list_images_with_meta(slug)
    if not images_with_meta:
        st.info("No images yet. Generate or upload some to get started.")
        return

    meta = _load_meta(slug)

    for idx, item in enumerate(images_with_meta):
        img_path: Path = item["path"]
        section: str = item["section"]
        pinned: bool = item["pinned"]
        favorite: bool = item["favorite"]
        filename = img_path.name

        cols = st.columns([3, 2])

        with cols[0]:
            try:
                st.image(str(img_path), use_container_width=True)
            except Exception:
                st.write(f"(Could not preview {img_path.name})")

            badges = []
            if pinned:
                badges.append("📌 Pinned")
            if favorite:
                badges.append("⭐ Favorite")
            badges.append(section.upper())
            st.caption(" • ".join(badges))
            st.code(str(img_path), language="bash")

        with cols[1]:
            st.markdown("**Actions**")

            if st.button("📌 Pin" if not pinned else "Unpin", key=f"pin_{idx}"):
                entry = meta.get(filename, {})
                entry["pinned"] = not pinned
                entry.setdefault("favorite", favorite)
                meta[filename] = entry
                _save_meta(slug, meta)
                st.rerun()

            if st.button(
                "⭐ Favorite" if not favorite else "Remove favorite",
                key=f"fav_{idx}",
            ):
                entry = meta.get(filename, {})
                entry["favorite"] = not favorite
                entry.setdefault("pinned", pinned)
                meta[filename] = entry
                _save_meta(slug, meta)
                st.rerun()

            firefly_url = make_firefly_edit_url(img_path)
            st.markdown(
                f"[Open in Adobe Firefly]({firefly_url})",
                unsafe_allow_html=True,
            )

            if st.button("🗑️ Delete", key=f"del_{idx}"):
                try:
                    img_path.unlink(missing_ok=True)
                except Exception as e:
                    st.error(f"Failed to delete {img_path.name}: {e}")
                if filename in meta:
                    del meta[filename]
                    _save_meta(slug, meta)
                st.rerun()

        st.markdown("---")


if __name__ == "__main__":
    main()
