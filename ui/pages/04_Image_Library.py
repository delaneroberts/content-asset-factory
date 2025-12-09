# ui/pages/04_Image_Library.py
from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import List

import shutil
import requests
import streamlit as st
from openai import OpenAI, OpenAIError
from PIL import Image  # NEW: for converting uploads to PNG

from caf_app.models import Campaign, ImageAsset
from caf_app.storage import load_campaign, save_campaign


# OpenAI client (uses OPENAI_API_KEY from .env)
client = OpenAI()

# Hugging Face config (NanoBanana + SDXL)
HF_API_KEY = os.getenv("HF_API_KEY")
HF_NANOBANANA_MODEL_ID = os.getenv(
    "HF_NANOBANANA_MODEL_ID", "REPLACE_WITH_NANOBANANA_MODEL_ID"
)
HF_SDXL_MODEL_ID = os.getenv(
    "HF_SDXL_MODEL_ID",
    "stabilityai/stable-diffusion-xl-base-1.0",
)

# Adobe Firefly config (OAuth server-to-server)
FIREFLY_CLIENT_ID = os.getenv("FIREFLY_SERVICES_CLIENT_ID")
FIREFLY_CLIENT_SECRET = os.getenv("FIREFLY_SERVICES_CLIENT_SECRET")

# Simple token cache for Firefly (access tokens valid ~24h)
_FIREFLY_ACCESS_TOKEN: str | None = None
_FIREFLY_TOKEN_EXPIRES_AT: float = 0.0


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


# ---- Firefly auth helper ---------------------------------------------------


def _get_firefly_access_token() -> str:
    """
    Get (and cache) an Adobe Firefly access token using client credentials.

    Uses FIREFLY_SERVICES_CLIENT_ID / FIREFLY_SERVICES_CLIENT_SECRET from env.
    """
    global _FIREFLY_ACCESS_TOKEN, _FIREFLY_TOKEN_EXPIRES_AT

    if _FIREFLY_ACCESS_TOKEN and time.time() < _FIREFLY_TOKEN_EXPIRES_AT - 300:
        return _FIREFLY_ACCESS_TOKEN

    if not FIREFLY_CLIENT_ID or not FIREFLY_CLIENT_SECRET:
        raise RuntimeError(
            "Adobe Firefly is not configured. "
            "Set FIREFLY_SERVICES_CLIENT_ID and FIREFLY_SERVICES_CLIENT_SECRET "
            "in your .env via Adobe Developer Console."
        )

    token_url = "https://ims-na1.adobelogin.com/ims/token/v3"
    payload = {
        "grant_type": "client_credentials",
        "client_id": FIREFLY_CLIENT_ID,
        "client_secret": FIREFLY_CLIENT_SECRET,
        "scope": (
            "openid,AdobeID,session,additional_info,read_organizations,"
            "firefly_api,ff_apis"
        ),
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    try:
        resp = requests.post(token_url, headers=headers, data=payload, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"Firefly auth request failed: {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(
            f"Firefly auth failed ({resp.status_code}): {resp.text[:500]}"
        )

    data = resp.json()
    access_token = data.get("access_token")
    if not access_token:
        raise RuntimeError("Firefly auth response missing access_token.")

    expires_in = data.get("expires_in", 24 * 3600)
    _FIREFLY_ACCESS_TOKEN = access_token
    _FIREFLY_TOKEN_EXPIRES_AT = time.time() + expires_in

    return access_token


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

        filename = (
            f"{campaign.slug}_{engine}_{len(campaign.image_assets) + idx + 1}.png"
        )
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


def _generate_images_from_pinned(
    campaign: Campaign,
    pinned_path: Path,
    prompt: str,
    n_images: int,
) -> List[ImageAsset]:
    """
    Use the pinned image as input to OpenAI's image edit API (gpt-image-1),
    so the model actually "sees" the pinned image and applies the prompt
    as an edit / variation.

    For now we always use gpt-image-1 here, regardless of the UI engine label.
    """
    images_dir = _campaign_images_dir(campaign.slug)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Call OpenAI image edit API with the pinned image as input
    try:
        with open(pinned_path, "rb") as f:
            result = client.images.edit(
                model="gpt-image-1",
                image=[f],
                prompt=prompt,
                n=n_images,
                size="1024x1024",
                output_format="png",
            )
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI image edit failed: {e}") from e

    if not getattr(result, "data", None):
        raise RuntimeError(
            "OpenAI did not return any image data for the edit request."
        )

    assets: List[ImageAsset] = []

    for idx, item in enumerate(result.data):
        b64_data = getattr(item, "b64_json", None)
        if not b64_data:
            continue

        image_bytes = base64.b64decode(b64_data)

        filename = (
            f"{campaign.slug}_pinned_{len(campaign.image_assets) + idx + 1}.png"
        )
        path = images_dir / filename

        with path.open("wb") as f_out:
            f_out.write(image_bytes)

        rel_path = path.relative_to(_project_root())

        assets.append(
            ImageAsset(
                path=str(rel_path),
                engine="pinned-template",  # label so you know it came from a template edit
                prompt=prompt,
            )
        )

    if not assets:
        raise RuntimeError(
            "OpenAI returned a response for the edit, but no usable image data."
        )

    return assets


def _generate_with_nanobanana_hf(
    campaign: Campaign,
    prompt: str,
    n_images: int,
) -> List[ImageAsset]:
    """
    Generate images using a NanoBanana (or similar) model hosted on Hugging Face.

    Uses:
      HF_API_KEY                -> Hugging Face access token
      HF_NANOBANANA_MODEL_ID    -> e.g. 'stabilityai/sdxl-turbo'
    """
    if not HF_API_KEY or not HF_NANOBANANA_MODEL_ID:
        raise RuntimeError(
            "NanoBanana (Hugging Face) is not configured. "
            "Set HF_API_KEY and HF_NANOBANANA_MODEL_ID in your environment."
        )

    api_url = (
        f"https://router.huggingface.co/hf-inference/models/{HF_NANOBANANA_MODEL_ID}"
    )
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "image/png",
    }

    images_dir = _campaign_images_dir(campaign.slug)
    images_dir.mkdir(parents=True, exist_ok=True)

    assets: List[ImageAsset] = []

    for idx in range(n_images):
        payload = {
            "inputs": prompt,
        }

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=120,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"NanoBanana HF request failed: {e}") from e

        if response.status_code != 200:
            raise RuntimeError(
                f"NanoBanana HF generation failed ({response.status_code}): "
                f"{response.text[:500]}"
            )

        # HF Inference API returns raw image bytes for image tasks
        image_bytes = response.content

        filename = (
            f"{campaign.slug}_nanobanana_{len(campaign.image_assets) + idx + 1}.png"
        )
        path = images_dir / filename

        with path.open("wb") as f:
            f.write(image_bytes)

        rel_path = path.relative_to(_project_root())

        assets.append(
            ImageAsset(
                path=str(rel_path),
                engine="nanobanana-hf",
                prompt=prompt,
            )
        )

    return assets


def _generate_with_sdxl_hf(
    campaign: Campaign,
    prompt: str,
    n_images: int,
) -> List[ImageAsset]:
    """
    Generate images using Stable Diffusion XL hosted on Hugging Face.

    Uses:
      HF_API_KEY       -> Hugging Face access token
      HF_SDXL_MODEL_ID -> e.g. 'stabilityai/stable-diffusion-xl-base-1.0'
    """
    if not HF_API_KEY or not HF_SDXL_MODEL_ID:
        raise RuntimeError(
            "Stable Diffusion XL (Hugging Face) is not configured. "
            "Set HF_API_KEY and HF_SDXL_MODEL_ID in your environment."
        )

    api_url = f"https://router.huggingface.co/hf-inference/models/{HF_SDXL_MODEL_ID}"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "image/png",
    }

    images_dir = _campaign_images_dir(campaign.slug)
    images_dir.mkdir(parents=True, exist_ok=True)

    assets: List[ImageAsset] = []

    for idx in range(n_images):
        payload = {"inputs": prompt}

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=120,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"SDXL HF request failed: {e}") from e

        if response.status_code != 200:
            raise RuntimeError(
                f"SDXL HF generation failed ({response.status_code}): "
                f"{response.text[:500]}"
            )

        image_bytes = response.content

        filename = (
            f"{campaign.slug}_sdxl_{len(campaign.image_assets) + idx + 1}.png"
        )
        path = images_dir / filename

        with path.open("wb") as f:
            f.write(image_bytes)

        rel_path = path.relative_to(_project_root())

        assets.append(
            ImageAsset(
                path=str(rel_path),
                engine="sdxl-hf",
                prompt=prompt,
            )
        )

    return assets


def _generate_with_firefly(
    campaign: Campaign,
    prompt: str,
    n_images: int,
    size: str,
) -> List[ImageAsset]:
    """
    Generate images using Adobe Firefly Text-to-Image API (v3 /images/generate).

    We:
      - Get an access token via IMS
      - Call Firefly with numVariations, prompt, and size
      - Download the returned image URLs and save them into the campaign folder.
    """
    token = _get_firefly_access_token()

    # Parse size like "1024x1024" -> width/height ints
    try:
        width_str, height_str = size.split("x")
        width = int(width_str)
        height = int(height_str)
    except Exception:
        width = height = 1024

    headers = {
        "X-Api-Key": FIREFLY_CLIENT_ID,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "numVariations": n_images,
        "prompt": prompt,
        "size": {"width": width, "height": height},
        "contentClass": "photo",  # could be "art" etc. later
    }

    try:
        resp = requests.post(
            "https://firefly-api.adobe.io/v3/images/generate",
            headers=headers,
            json=payload,
            timeout=120,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Firefly request failed: {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(
            f"Firefly generation failed ({resp.status_code}): {resp.text[:500]}"
        )

    data = resp.json()
    outputs = data.get("outputs") or []
    if not outputs:
        raise RuntimeError("Firefly returned no outputs for this request.")

    images_dir = _campaign_images_dir(campaign.slug)
    images_dir.mkdir(parents=True, exist_ok=True)

    assets: List[ImageAsset] = []

    # Each output has image.url; download them
    for idx, out in enumerate(outputs[:n_images]):
        img_info = out.get("image") or {}
        url = img_info.get("url")
        if not url:
            continue

        try:
            img_resp = requests.get(url, timeout=120)
        except requests.RequestException:
            continue

        if img_resp.status_code != 200:
            continue

        filename = f"{campaign.slug}_firefly_{len(campaign.image_assets) + idx + 1}.png"
        path = images_dir / filename
        with path.open("wb") as f:
            f.write(img_resp.content)

        rel_path = path.relative_to(_project_root())
        assets.append(
            ImageAsset(
                path=str(rel_path),
                engine="firefly",
                prompt=prompt,
            )
        )

    if not assets:
        raise RuntimeError(
            "Firefly returned outputs but no downloadable images. "
            "Check Firefly permissions and response in logs."
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
      - "sdxl-hf"           -> Stable Diffusion XL on Hugging Face
      - "firefly"           -> Adobe Firefly
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
    elif engine == "sdxl-hf":
        return _generate_with_sdxl_hf(
            campaign=campaign,
            prompt=prompt,
            n_images=n_images,
        )
    elif engine == "firefly":
        return _generate_with_firefly(
            campaign=campaign,
            prompt=prompt,
            n_images=n_images,
            size=size,
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

        # Optional: use pinned image as template
        use_pinned_template = False
        if pinned_path:
            st.info(
                "A pinned image is available. You can use it as a base/template "
                "for the new images."
            )
            use_pinned_template = st.checkbox(
                "Use pinned image as template",
                value=False,
                help=(
                    "When enabled, new images will be derived from the pinned image "
                    "instead of purely from the text prompt."
                ),
            )

    with col2:
        engine = st.selectbox(
            "Engine",
            options=["dall-e-3", "dall-e-2", "nanobanana-hf", "sdxl-hf", "firefly"],
            index=0,
            format_func=lambda v: {
                "dall-e-3": "OpenAI – DALL·E 3",
                "dall-e-2": "OpenAI – DALL·E 2",
                "nanobanana-hf": "NanoBanana – Hugging Face",
                "sdxl-hf": "Stable Diffusion XL – Hugging Face",
                "firefly": "Adobe Firefly",
            }[v],
            help="Choose which backend to use for image generation.",
        )

        # Per-engine limits (matches your prior behavior, with SDXL same as NanoBanana)
        if engine == "dall-e-3":
            max_images = 2
        elif engine == "dall-e-2":
            max_images = 8
        elif engine == "firefly":
            max_images = 4
        else:  # nanobanana-hf or sdxl-hf
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
        elif engine == "firefly":
            # Firefly uses separate width/height fields, but we keep the same string format
            size_options = ["1024x1024", "2048x2048"]
        else:  # nanobanana-hf or sdxl-hf
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
                    # If user chose to use the pinned image as a template and one exists,
                    # route through the pinned-image helper.
                    if pinned_path and use_pinned_template:
                        new_assets = _generate_images_from_pinned(
                            campaign=campaign,
                            pinned_path=Path(pinned_path),
                            prompt=prompt.strip(),
                            n_images=int(n_images),
                        )
                    else:
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

    # 1) Clean up any assets whose files are missing on disk
    all_assets: List[ImageAsset] = campaign.image_assets
    valid_assets: List[ImageAsset] = []
    missing_count = 0

    for asset in all_assets:
        img_path = _project_root() / asset.path
        if img_path.exists():
            valid_assets.append(asset)
        else:
            missing_count += 1

    if missing_count:
        campaign.image_assets = valid_assets
        save_campaign(campaign)
        st.info(f"Cleaned up {missing_count} image entries with missing files.")

    assets = valid_assets

    # 2) Apply "favorites only" filter
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
            st.image(str(image_path), use_container_width=True)

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
                    img_path = Path(asset.path)
                    if not img_path.is_absolute():
                        try:
                            img_path = _project_root() / img_path
                        except NameError:
                            img_path = Path.cwd() / img_path

                    try:
                        img_path.unlink(missing_ok=True)
                    except Exception as e:
                        st.error(f"Failed to delete file: {e}")
                    else:
                        campaign.image_assets = [
                            a for a in campaign.image_assets
                            if a.id != asset.id
                        ]
                        save_campaign(campaign)
                        st.rerun()


if __name__ == "__main__":
    main()
