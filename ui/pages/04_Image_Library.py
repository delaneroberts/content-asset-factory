# ui/pages/04_Image_Library.py
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import List

import requests
import streamlit as st
from openai import OpenAI, OpenAIError

from caf_app.models import Campaign, ImageAsset
from caf_app.storage import load_campaign, save_campaign


# OpenAI client (uses OPENAI_API_KEY from .env)
client = OpenAI()

# Hugging Face NanoBanana config
HF_API_KEY = os.getenv("HF_API_KEY")
# TODO: set this to your actual NanoBanana model id from Hugging Face,
# e.g. "your-username/your-nanobanana-model"
HF_NANOBANANA_MODEL_ID = os.getenv("HF_NANOBANANA_MODEL_ID", "REPLACE_WITH_NANOBANANA_MODEL_ID")


# ---- Helpers ---------------------------------------------------------------


def _get_current_slug() -> str | None:
    return st.session_state.get("current_campaign_slug")


def _project_root() -> Path:
    # This file is at <root>/ui/pages/04_Image_Library.py
    return Path(__file__).resolve().parents[2]


def _campaign_images_dir(slug: str) -> Path:
    root = _project_root()
    directory = root / "generated_images" / slug
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _default_image_prompt(campaign: Campaign) -> str:
    parts = []
    if campaign.product_name:
        parts.append(f"{campaign.product_name}")
    if campaign.description:
        parts.append(campaign.description)
    if campaign.audience:
        parts.append(f"For: {campaign.audience}")
    if campaign.tone:
        parts.append(f"Tone: {campaign.tone}")

    base = " ".join(parts) or "A clean product hero shot on a simple background."

    return (
        f"{base}. Studio-quality product hero image, high detail, soft lighting, "
        "clean background, no text or logos."
    )

# ---- Engine-specific generators -------------------------------------------


def _generate_with_openai(
    campaign: Campaign,
    prompt: str,
    engine: str,
    n_images: int,
    size: str,
) -> List[ImageAsset]:
    """
    Generate images using OpenAI's current image model (gpt-image-1).

    We keep the 'engine' parameter for UI labeling and metadata,
    but under the hood everything goes through gpt-image-1.
    """
    images_dir = _campaign_images_dir(campaign.slug)

    try:
        # This matches your working testme.py call
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=n_images,
            size=size,
            output_format="png",
        )
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI image generation failed: {e}") from e


    # Decode the images and write them into the campaign's images dir
    assets: List[ImageAsset] = []

    if not getattr(response, "data", None):
        raise RuntimeError(
            "OpenAI did not return any image data. "
            "This usually means your project does not have image model access, "
            "or the request violated a safety / quota limit."
        )

    for idx, item in enumerate(response.data):
        b64_data = getattr(item, "b64_json", None)
        if not b64_data:
            continue

        image_bytes = base64.b64decode(b64_data)

        filename = f"{campaign.slug}_{engine}_{len(campaign.image_assets) + idx + 1}.png"
        path = images_dir / filename

        with path.open("wb") as f:
            f.write(image_bytes)

        rel_path = path.relative_to(_project_root())

        assets.append(
            ImageAsset(
                path=str(rel_path),
                engine=engine,  # still records "dall-e-3" or "dall-e-2" for display
                prompt=prompt,
            )
        )

    if not assets:
        raise RuntimeError(
            "OpenAI returned a response but no usable image data. "
            "Check logs and model access."
        )

    return assets



def _generate_image_files(
    campaign: Campaign,
    prompt: str,
    engine: str,
    n_images: int,
    size: str = "1024x1024",
) -> List[ImageAsset]:
    """
    Unified entry point: route to the appropriate backend based on engine.
    engine:
      - "dall-e-3"          -> OpenAI DALL·E 3
      - "dall-e-2"          -> OpenAI DALL·E 2
      - "nanobanana-hf"     -> NanoBanana on Hugging Face
    """
    if engine in ("dall-e-3", "dall-e-2"):
        return _generate_with_openai(
            campaign=campaign,
            prompt=prompt,
            engine=engine,
            n_images=n_images,
            size=size,
        )
    elif engine == "nanobanana-hf":
        return _generate_with_nanobanana_hf(
            campaign=campaign,
            prompt=prompt,
            n_images=n_images,
        )
    else:
        raise RuntimeError(f"Unknown image engine: {engine}")


# ---- Main UI ---------------------------------------------------------------

def main() -> None:
    st.title("Image Library")

    slug = _get_current_slug()
    if not slug:
        st.warning(
            "No campaign selected. Go to **Dashboard** first and create or open a campaign."
        )
        return

    campaign = load_campaign(slug)
    if campaign is None:
        st.error(
            f"Could not load campaign with slug `{slug}`. "
            "It may have been deleted or the file is corrupted."
        )
        return

    st.caption(f"Campaign: **{campaign.name}** (`{campaign.slug}`)")

    # --- Pinned image (sidebar) ---------------------------------------------
    pinned_path = st.session_state.get("pinned_image")
    if pinned_path:
        pinned_file = Path(pinned_path)
        if pinned_file.exists():
            st.sidebar.subheader("Pinned image")
            st.sidebar.image(str(pinned_file), use_container_width=True)
            if st.sidebar.button("Unpin image"):
                del st.session_state["pinned_image"]
                st.rerun()
        else:
            # Clear stale pinned path
            st.session_state.pop("pinned_image", None)

    # --- Generation controls -------------------------------------------------
    st.markdown("### Generate new images")

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        default_prompt = _default_image_prompt(campaign)
        prompt = st.text_area(
            "Image prompt",
            value=default_prompt,
            height=120,
        )

    with col2:
        engine = st.selectbox(
            "Engine",
            options=["dall-e-3", "dall-e-2", "nanobanana-hf"],
            index=0,
            format_func=lambda v: {
                "dall-e-3": "OpenAI – DALL·E 3",
                "dall-e-2": "OpenAI – DALL·E 2",
                "nanobanana-hf": "NanoBanana – Hugging Face",
            }[v],
            help="Choose which backend to use for image generation.",
        )

        # Per-engine limits
        if engine == "dall-e-3":
            max_images = 4
        elif engine == "dall-e-2":
            max_images = 8
        else:  # nanobanana-hf
            max_images = 4  # conservative default

        n_images = st.number_input(
            "Count",
            min_value=1,
            max_value=max_images,
            value=min(4, max_images),
            step=1,
        )

    with col3:
        # Size options depend on engine
        if engine == "dall-e-3":
            size_options = ["1024x1024", "1024x1792", "1792x1024"]
        elif engine == "dall-e-2":
            size_options = ["1024x1024", "512x512"]
        else:  # nanobanana-hf
            # Most SD-based models default to 512x512 or 1024x1024; we keep it simple for now
            size_options = ["1024x1024"]

        size = st.selectbox(
            "Size",
            options=size_options,
            index=0,
        )

    if st.button("Generate images", type="primary"):
        if not prompt.strip():
            st.warning("Please enter an image prompt.")
        else:
            try:
                with st.spinner("Generating images..."):
                    new_assets = _generate_image_files(
                        campaign=campaign,
                        prompt=prompt.strip(),
                        engine=engine,
                        n_images=int(n_images),
                        size=size,
                    )
                    campaign.image_assets.extend(new_assets)
                    save_campaign(campaign)

                st.success(f"Added {len(new_assets)} new images to this campaign.")
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))

    st.markdown("---")

    # --- Existing images -----------------------------------------------------
    st.markdown("### Existing images for this campaign")

    if not campaign.image_assets:
        st.write("No images generated yet. Use the controls above to generate some.")
        return

    show_only_favorites = st.checkbox("Show only favorites", value=False)

    assets = campaign.image_assets
    if show_only_favorites:
        assets = [a for a in assets if a.is_favorite]

    if not assets:
        st.write("No images match this filter yet.")
        return

    cols = st.columns(3)

    for idx, asset in enumerate(assets):
        col = cols[idx % 3]
        with col:
            image_path = _project_root() / asset.path
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.warning(f"Missing file: {asset.path}")

            st.caption(f"Engine: `{asset.engine}`")
            if asset.prompt:
                with st.expander("Prompt"):
                    st.write(asset.prompt)

            # --- Favorite + Pin + Delete controls ---
            col_fav, col_pin, col_del = st.columns([3, 2, 1])

            # Favorite / unfavorite
            with col_fav:
                fav_label = "★ Unfavorite" if asset.is_favorite else "☆ Favorite"
                if st.button(fav_label, key=f"fav_img_{asset.id}"):
                    asset.is_favorite = not asset.is_favorite
                    save_campaign(campaign)
                    st.rerun()

            # 📌 Pin image
            with col_pin:
                if st.button("📌 Pin", key=f"pin_img_{asset.id}"):
                    st.session_state["pinned_image"] = str(image_path)
                    st.success("Pinned this image for reuse.")
                    st.rerun()

            # Delete image
            with col_del:
                if st.button("🗑", key=f"del_img_{asset.id}", help="Delete this image"):
                    # Resolve the image file path on disk
                    img_path = Path(asset.path)
                    if not img_path.is_absolute():
                        # If you have a _project_root() helper earlier in this file, use it:
                        try:
                            img_path = _project_root() / img_path  # type: ignore[name-defined]
                        except NameError:
                            # Fallback: assume asset.path is relative to current working dir
                            img_path = Path.cwd() / img_path

                    # 1) Delete the image file
                    try:
                        img_path.unlink(missing_ok=True)
                    except Exception as e:
                        st.error(f"Failed to delete file: {e}")
                    else:
                        # 2) Remove from the campaign's image_assets list
                        campaign.image_assets = [
                            a for a in campaign.image_assets
                            if a.id != asset.id
                        ]
                        # 3) Persist the change
                        save_campaign(campaign)
                        # 4) Refresh the page so the image disappears
                        st.rerun()



if __name__ == "__main__":
    main()
