# ui/05_Concept_Builder.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List
from datetime import datetime

import streamlit as st

from caf_app.models import Campaign, Concept, ImageAsset, TextAsset
from caf_app.storage import load_campaign, save_campaign




# ---- Helpers ---------------------------------------------------------------


def _get_current_slug() -> str | None:
    return st.session_state.get("current_campaign_slug")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_image(asset_list: List[ImageAsset], asset_id: str) -> Optional[ImageAsset]:
    for a in asset_list:
        if a.id == asset_id:
            return a
    return None


def _text_options(text_assets: List[TextAsset], kind: str) -> List[TextAsset]:
    return [t for t in text_assets if t.kind == kind]


# ---- Main UI ---------------------------------------------------------------


def main() -> None:
    st.title("Concept Builder")

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

    if not campaign.image_assets:
        st.info("You don't have any images yet. Generate some in **Image Library** first.")
        return

    if not campaign.text_assets:
        st.info("You don't have any copy yet. Generate some in **Text Library** first.")
        return

    # --- Create new concept --------------------------------------------------
    st.markdown("### Create a new concept")

    col_img, col_txt = st.columns([2, 3])

    # Image selection
    with col_img:
        st.subheader("Choose image")

        # Offer only favorites by default, but allow "all"
        use_only_fav_images = st.checkbox("Use only favorite images", value=True)

        images = (
            [img for img in campaign.image_assets if img.is_favorite]
            if use_only_fav_images
            else campaign.image_assets
        )
        if not images:
            st.warning(
                "No images match this filter. Uncheck 'Use only favorite images' or "
                "favorite some images in the Image Library."
            )
            images = campaign.image_assets

        image_labels = [
            f"{idx+1}. {img.engine} – {img.path.split('/')[-1]}"
            for idx, img in enumerate(images)
        ]
        selected_idx = st.selectbox(
            "Image",
            options=list(range(len(images))),
            format_func=lambda i: image_labels[i],
        )
        selected_image = images[selected_idx]

        img_path = _project_root() / selected_image.path
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.error(f"Image file not found: {selected_image.path}")

    # Text selection
    with col_txt:
        st.subheader("Choose text")

        use_only_fav_text = st.checkbox("Use only favorite copy", value=True)

        def filter_text(kind: str) -> List[TextAsset]:
            assets = _text_options(campaign.text_assets, kind)
            if use_only_fav_text:
                assets = [a for a in assets if a.is_favorite]
            return assets

        tagline_options = filter_text("tagline")
        header_options = filter_text("header")
        subheader_options = filter_text("subheader")
        body_options = filter_text("body")

        def label_text(t: TextAsset) -> str:
            snippet = t.content.strip().replace("\n", " ")
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            return snippet

        # Each selectbox includes a "None" option
        tagline_choice = st.selectbox(
            "Tagline (optional)",
            options=["None"] + tagline_options,
            format_func=lambda x: "None" if x == "None" else label_text(x),
        )
        header_choice = st.selectbox(
            "Header (optional)",
            options=["None"] + header_options,
            format_func=lambda x: "None" if x == "None" else label_text(x),
        )
        subheader_choice = st.selectbox(
            "Subheader (optional)",
            options=["None"] + subheader_options,
            format_func=lambda x: "None" if x == "None" else label_text(x),
        )
        body_choice = st.selectbox(
            "Body copy (optional)",
            options=["None"] + body_options,
            format_func=lambda x: "None" if x == "None" else label_text(x),
        )

        tagline_text = None if tagline_choice == "None" else tagline_choice.content
        header_text = None if header_choice == "None" else header_choice.content
        subheader_text = (
            None if subheader_choice == "None" else subheader_choice.content
        )
        body_text = None if body_choice == "None" else body_choice.content

        st.markdown("#### Preview text")

        if tagline_text:
            st.markdown(f"**Tagline:** {tagline_text}")
        if header_text:
            st.markdown(f"**Header:** {header_text}")
        if subheader_text:
            st.markdown(f"**Subheader:** {subheader_text}")
        if body_text:
            st.markdown(f"**Body:** {body_text}")

        if st.button("Create concept", type="primary"):
            if not any([tagline_text, header_text, subheader_text, body_text]):
                st.warning("Please select at least one piece of text.")
            else:
                concept = Concept(
                    image_asset_id=selected_image.id,
                    tagline=tagline_text,
                    header=header_text,
                    subheader=subheader_text,
                    body=body_text,
                )
                campaign.concepts.append(concept)
                save_campaign(campaign)
                st.success("Concept created.")
                st.rerun()

    st.markdown("---")

    # --- Existing concepts ---------------------------------------------------
    st.markdown("### Existing concepts")

    if not campaign.concepts:
        st.write("No concepts yet. Create one above.")
        return

    show_only_favorites = st.checkbox("Show only favorite concepts", value=False)

    concepts = (
        [c for c in campaign.concepts if c.is_favorite]
        if show_only_favorites
        else campaign.concepts
    )

    if not concepts:
        st.write("No concepts match this filter yet.")
        return

    for concept in concepts:
        with st.container(border=True):
            cols = st.columns([2, 3])

            # Image preview
            with cols[0]:
                image_asset = _find_image(campaign.image_assets, concept.image_asset_id)
                if image_asset:
                    img_path = _project_root() / image_asset.path
                    if img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.error(f"Image file not found: {image_asset.path}")
                else:
                    st.error("Referenced image not found in campaign.image_assets")

            # Text + controls
            with cols[1]:
                st.markdown("**Text**")

                tagline_val = st.text_input(
                    "Tagline",
                    value=concept.tagline or "",
                    key=f"tagline_{concept.id}",
                )
                header_val = st.text_input(
                    "Header",
                    value=concept.header or "",
                    key=f"header_{concept.id}",
                )
                subheader_val = st.text_area(
                    "Subheader",
                    value=concept.subheader or "",
                    key=f"subheader_{concept.id}",
                    height=60,
                )
                body_val = st.text_area(
                    "Body",
                    value=concept.body or "",
                    key=f"body_{concept.id}",
                    height=100,
                )

                btn_cols = st.columns([2, 2, 2])
                with btn_cols[0]:
                    if st.button("Save changes", key=f"save_{concept.id}"):
                        concept.tagline = tagline_val.strip() or None
                        concept.header = header_val.strip() or None
                        concept.subheader = subheader_val.strip() or None
                        concept.body = body_val.strip() or None
                        concept.updated_at = datetime.utcnow()
                        save_campaign(campaign)
                        st.success("Concept updated.")
                        st.rerun()

                with btn_cols[1]:
                    fav_label = "★ Unfavorite" if concept.is_favorite else "☆ Favorite"
                    if st.button(fav_label, key=f"fav_concept_{concept.id}"):
                        concept.is_favorite = not concept.is_favorite
                        save_campaign(campaign)
                        st.rerun()

                with btn_cols[2]:
                    if st.button(
                        "Delete",
                        key=f"delete_{concept.id}",
                        help="Remove this concept from the campaign.",
                    ):
                        campaign.concepts = [
                            c for c in campaign.concepts if c.id != concept.id
                        ]
                        save_campaign(campaign)
                        st.warning("Concept deleted.")
                        st.rerun()


if __name__ == "__main__":
    main()
