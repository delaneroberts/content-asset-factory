# ui/03_Text_Library.py
from __future__ import annotations

from typing import List

import streamlit as st
from openai import OpenAI

from caf_app.models import Campaign, TextAsset, TextAssetKind
from caf_app.storage import load_campaign, save_campaign


client = OpenAI()  # Uses OPENAI_API_KEY from your environment


def _get_current_slug() -> str | None:
    return st.session_state.get("current_campaign_slug")


def _build_base_prompt(campaign: Campaign, kind: TextAssetKind, n_variants: int) -> str:
    kind_label = {
        "tagline": "short punchy taglines (max 8 words each)",
        "header": "hero headers (max 12 words each)",
        "subheader": "supporting subheaders (1 short sentence each)",
        "body": "short body copy paragraphs (2–3 sentences each)",
    }[kind]

    brief_parts = []
    if campaign.product_name:
        brief_parts.append(f"Product: {campaign.product_name}")
    if campaign.description:
        brief_parts.append(f"Description: {campaign.description}")
    if campaign.audience:
        brief_parts.append(f"Target audience: {campaign.audience}")
    if campaign.tone:
        brief_parts.append(f"Tone/style: {campaign.tone}")
    if campaign.notes:
        brief_parts.append(f"Notes: {campaign.notes}")

    brief_text = "\n".join(brief_parts) or "No additional brief provided."

    return (
        f"You are a creative marketing copywriter.\n\n"
        f"Campaign brief:\n{brief_text}\n\n"
        f"Generate {n_variants} distinct {kind_label} for this campaign.\n"
        f"Return them as a simple numbered list, one per line."
    )


def _generate_text_variants(
    campaign: Campaign,
    kind: TextAssetKind,
    n_variants: int,
    extra_instructions: str | None = None,
) -> List[TextAsset]:
    base_prompt = _build_base_prompt(campaign, kind, n_variants)
    if extra_instructions:
        base_prompt += f"\n\nAdditional instructions: {extra_instructions.strip()}"

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You write concise, high-impact marketing copy.",
            },
            {
                "role": "user",
                "content": base_prompt,
            },
        ],
        temperature=0.9,
        max_tokens=512,
    )

    content = response.choices[0].message.content or ""
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    variants: List[TextAsset] = []
    for line in lines:
        # Strip leading numbers like "1. " or "1)"
        if line[0].isdigit():
            i = 1
            while i < len(line) and (line[i].isdigit() or line[i] in ". )"):
                i += 1
            line = line[i:].strip()
        if not line:
            continue
        variants.append(
            TextAsset(
                kind=kind,
                content=line,
                source="ai",
                prompt=base_prompt,
                model_name="gpt-4.1-mini",
            )
        )

    return variants[:n_variants]


def main() -> None:
    st.title("Text Library")

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
    st.markdown("### Generate new copy")

    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        kind_label = st.selectbox(
            "Type of copy",
            options=[
                ("tagline", "Taglines"),
                ("header", "Hero headers"),
                ("subheader", "Subheaders"),
                ("body", "Short body copy"),
            ],
            format_func=lambda x: x[1],
        )
        kind: TextAssetKind = kind_label[0]  # "tagline", etc.

    with col2:
        n_variants = st.number_input(
            "Number of variants",
            min_value=1,
            max_value=20,
            value=6,
            step=1,
        )

    with col3:
        extra = st.text_input(
            "Optional tweak instructions",
            placeholder="e.g., make them funnier, more premium, more urgent...",
        )

    if st.button("Generate copy", type="primary"):
        with st.spinner("Generating copy variants..."):
            new_assets = _generate_text_variants(
                campaign=campaign,
                kind=kind,
                n_variants=n_variants,
                extra_instructions=extra or None,
            )
            campaign.text_assets.extend(new_assets)
            save_campaign(campaign)

        st.success(f"Added {len(new_assets)} new {kind} variants to this campaign.")
        st.rerun()


    st.markdown("---")

    # --- Existing text assets -----------------------------------------------
    st.markdown("### Existing copy for this campaign")

    if not campaign.text_assets:
        st.write("No copy generated yet. Use the controls above to generate some.")
        return

    kind_filter = st.selectbox(
        "Filter by type",
        options=["all", "tagline", "header", "subheader", "body"],
        index=0,
    )

    show_only_favorites = st.checkbox("Show only favorites", value=False)

    filtered = campaign.text_assets
    if kind_filter != "all":
        filtered = [t for t in filtered if t.kind == kind_filter]
    if show_only_favorites:
        filtered = [t for t in filtered if t.is_favorite]

    if not filtered:
        st.write("No copy matches this filter yet.")
        return

    for asset in filtered:
        with st.container(border=True):
            top_cols = st.columns([6, 2])

            with top_cols[0]:
                st.markdown(f"**Type:** `{asset.kind}`")
                st.caption(
                    f"Created: {asset.created_at.strftime('%Y-%m-%d %H:%M')} "
                    f"{'(★ favorite)' if asset.is_favorite else ''}"
                )

            with top_cols[1]:
                fav_label = "Unfavorite" if asset.is_favorite else "Favorite"
                if st.button(
                    fav_label,
                    key=f"fav_{asset.id}",
                ):
                    asset.is_favorite = not asset.is_favorite
                    save_campaign(campaign)
                    st.rerun()

            st.markdown("---")
            st.write(asset.content)


if __name__ == "__main__":
    main()
