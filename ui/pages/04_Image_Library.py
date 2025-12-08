# ui/pages/04_Image_Library.py
from __future__ import annotations

import base64
from pathlib import Path
from typing import List

import streamlit as st
from openai import OpenAI, OpenAIError

from caf_app.models import Campaign, ImageAsset
from caf_app.storage import load_campaign, save_campaign


client = OpenAI()  # Uses OPENAI_API_KEY from your environment


# ---- Helpers ---------------------------------------------------------------


def _get_current_slug() -> str | None:
    return st.session_state.get("current_campaign_slug")


def _project_root() -> Path:
    # This file is at <root>/ui/pages/04_Image_Library.py
    # So project root is two levels up
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


def _generate_image_files(
    campaign: Campaign,
    prompt: str,
    engine: str,
    n_images: int,
    size: str = "1024x1024",
) -> List[ImageAsset]:
    """
    Use OpenAI's image generation to create n_images and save to disk.
    DALL·E 3: only supports n=1 and does not accept "n".
    DALL·E 2: supports multiple images + "n".
    """
    images_dir = _campaign_images_dir(campaign.slug)

    # Backend guard: enforce DALL·E 3 limit
    if engine == "dall-e-3" and n_images > 1:
        n_images = 1

    try:
        if engine == "dall-e-3":
            # DALL·E 3 request format (no n)
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
            )
        else:
            # DALL·E 2 request format (supports n)
            response = client.images.generate(
                model="dall-e-2",
                prompt=prompt,
                n=n_images,
                size=size,
            )

    except OpenAIError as e:
        # Bubble a clean error up to Streamlit
        raise RuntimeError(f"Error from OpenAI image API: {e}") from e

    assets: List[ImageAsset] = []

    # response.data is a list of image results with b64_json
    for idx, item in enumerate(response.data):
        b64_data = getattr(item, "b64_json", None)
        if not b64_data:
            continue

        image_bytes = base64.b64decode(b64_data)

        filename = f"{campaign.slug}_{engine}_{len(campaign.image_assets) + idx + 1}.png"
        path = images_dir / filename

        with path.open("wb") as f:
            f.write(image_bytes)

        # Store path relative to project root for portability
        rel_path = path.relative_to(_project_root())

        assets.append(
            ImageAsset(
                path=str(rel_path),
                engine=engine,
                prompt=prompt,
            )
        )

    return assets


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
            options=["dall-e-3", "dall-e-2"],
            index=0,
            help="OpenAI image engines. 'dall-e-3' only supports 1 image at a time.",
        )

        max_images = 1 if engine == "dall-e-3" else 8
        n_images = st.number_input(
            "Count",
            min_value=1,
            max_value=max_images,
            value=min(3, max_images),
            step=1,
        )

    with col3:
        if engine == "dall-e-3":
            size_options = ["1024x1024", "1024x1792", "1792x1024"]
        else:
            # DALL·E 2 supports these sizes in the new API
            size_options = ["1024x1024", "512x512"]  # 512x512 is supported for dalle2

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

            fav_label = "★ Unfavorite" if asset.is_favorite else "☆ Favorite"
            if st.button(fav_label, key=f"fav_img_{asset.id}"):
                asset.is_favorite = not asset.is_favorite
                save_campaign(campaign)
                st.rerun()


if __name__ == "__main__":
    main()
