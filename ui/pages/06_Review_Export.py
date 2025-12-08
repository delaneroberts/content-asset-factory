# ui/pages/06_Review_Export.py
from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import streamlit as st

from caf_app.models import Campaign, Concept, ImageAsset
from caf_app.storage import load_campaign



# ---- Helpers ---------------------------------------------------------------


def _get_current_slug() -> str | None:
    return st.session_state.get("current_campaign_slug")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]  # .../ui/pages -> project root


def _find_image(asset_list: List[ImageAsset], asset_id: str) -> Optional[ImageAsset]:
    for a in asset_list:
        if a.id == asset_id:
            return a
    return None


def _build_markdown_summary(
    campaign: Campaign,
    concepts: List[Concept],
    images_by_id: dict[str, ImageAsset],
) -> str:
    """Build a markdown summary of the campaign + selected concepts."""
    lines: List[str] = []

    lines.append(f"# Campaign: {campaign.name}")
    lines.append("")
    lines.append(f"- Slug: `{campaign.slug}`")
    lines.append(f"- Created: {campaign.created_at.isoformat()}")
    lines.append(f"- Updated: {campaign.updated_at.isoformat()}")
    lines.append("")

    lines.append("## Brief")
    brief_parts = []
    if campaign.product_name:
        brief_parts.append(f"**Product:** {campaign.product_name}")
    if campaign.description:
        brief_parts.append(f"**Description:** {campaign.description}")
    if campaign.audience:
        brief_parts.append(f"**Audience:** {campaign.audience}")
    if campaign.tone:
        brief_parts.append(f"**Tone:** {campaign.tone}")
    if campaign.notes:
        brief_parts.append(f"**Notes:** {campaign.notes}")

    if brief_parts:
        lines.extend(brief_parts)
    else:
        lines.append("_No brief details provided._")

    lines.append("")
    lines.append("## Concepts")
    lines.append("")

    if not concepts:
        lines.append("_No concepts selected._")
    else:
        for idx, concept in enumerate(concepts, start=1):
            lines.append(f"### Concept {idx}")
            img = images_by_id.get(concept.image_asset_id)
            if img:
                filename = Path(img.path).name
                lines.append(f"- Image file: `images/{filename}`")
                lines.append(f"- Engine: `{img.engine}`")
            else:
                lines.append("- Image: _missing_")

            if concept.tagline:
                lines.append(f"- **Tagline:** {concept.tagline}")
            if concept.header:
                lines.append(f"- **Header:** {concept.header}")
            if concept.subheader:
                lines.append(f"- **Subheader:** {concept.subheader}")
            if concept.body:
                lines.append("")
                lines.append("**Body:**")
                lines.append("")
                lines.append(concept.body)

            lines.append("")

    return "\n".join(lines)


def _build_export_zip(
    campaign: Campaign,
    concepts: List[Concept],
) -> bytes:
    """Create an in-memory ZIP containing images + markdown summary."""
    project_root = _project_root()

    # Map image IDs to assets once
    images_by_id: dict[str, ImageAsset] = {
        img.id: img for img in campaign.image_assets
    }

    # Prepare in-memory buffer
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add images
        for concept in concepts:
            img = images_by_id.get(concept.image_asset_id)
            if not img:
                continue
            img_path = project_root / img.path
            if not img_path.exists():
                continue
            arcname = f"images/{img_path.name}"
            zf.write(img_path, arcname=arcname)

        # Add markdown summary
        summary_md = _build_markdown_summary(campaign, concepts, images_by_id)
        zf.writestr("campaign_summary.md", summary_md)

    buffer.seek(0)
    return buffer.getvalue()


# ---- Main UI ---------------------------------------------------------------


def main() -> None:
    st.title("Review & Export")

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

    if not campaign.concepts:
        st.info("You don't have any concepts yet. Build some in **Concept Builder** first.")
        return

    st.markdown("### Select concepts for export")

    show_only_favorites = st.checkbox("Show only favorite concepts", value=False)

    concepts = (
        [c for c in campaign.concepts if c.is_favorite]
        if show_only_favorites
        else campaign.concepts
    )

    if not concepts:
        st.write("No concepts match this filter yet.")
        return

    # Build a list of IDs for selection
    concept_ids = [c.id for c in concepts]

    # Default: select all
    if "export_selected_concepts" not in st.session_state:
        st.session_state.export_selected_concepts = set(concept_ids)

    # Buttons to select/deselect all
    col_all, col_none = st.columns(2)
    with col_all:
        if st.button("Select all"):
            st.session_state.export_selected_concepts = set(concept_ids)
    with col_none:
        if st.button("Select none"):
            st.session_state.export_selected_concepts = set()

    selected_ids: set[str] = st.session_state.export_selected_concepts

    st.markdown("---")

    # Display each concept with a checkbox
    for concept in concepts:
        with st.container(border=True):
            cols = st.columns([1, 2, 3])

            # Checkbox
            with cols[0]:
                checked = concept.id in selected_ids
                new_val = st.checkbox(
                    "Include",
                    value=checked,
                    key=f"include_{concept.id}",
                )
                if new_val and concept.id not in selected_ids:
                    selected_ids.add(concept.id)
                elif not new_val and concept.id in selected_ids:
                    selected_ids.remove(concept.id)

            # Image
            with cols[1]:
                image_asset = _find_image(campaign.image_assets, concept.image_asset_id)
                if image_asset:
                    img_path = _project_root() / image_asset.path
                    if img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.error(f"Image file not found: {image_asset.path}")
                else:
                    st.error("Referenced image not found in campaign.image_assets")

            # Text
            with cols[2]:
                st.markdown("**Text**")
                if concept.tagline:
                    st.markdown(f"**Tagline:** {concept.tagline}")
                if concept.header:
                    st.markdown(f"**Header:** {concept.header}")
                if concept.subheader:
                    st.markdown(f"**Subheader:** {concept.subheader}")
                if concept.body:
                    st.markdown("**Body:**")
                    st.write(concept.body)

                st.caption(
                    f"Created: {concept.created_at.strftime('%Y-%m-%d %H:%M')} "
                    f"{'(★ favorite)' if concept.is_favorite else ''}"
                )

    # Persist updated selection
    st.session_state.export_selected_concepts = selected_ids

    st.markdown("---")

    selected_concepts = [c for c in concepts if c.id in selected_ids]

    if not selected_concepts:
        st.warning("No concepts selected. Check at least one to enable export.")
        return

    st.markdown(f"**{len(selected_concepts)}** concepts selected for export.")

    if st.button("Build export ZIP", type="primary"):
        with st.spinner("Building ZIP package..."):
            zip_bytes = _build_export_zip(campaign, selected_concepts)

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filename = f"{campaign.slug}_export_{timestamp}.zip"

        st.success("Export package created.")
        st.download_button(
            "Download ZIP",
            data=zip_bytes,
            file_name=filename,
            mime="application/zip",
        )


if __name__ == "__main__":
    main()
