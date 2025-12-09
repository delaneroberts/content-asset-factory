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
        "You are a creative marketing copywriter.\n\n"
        f"Campaign brief:\n{brief_text}\n\n"
        f"Generate {n_variants} distinct {kind_label} for this campaign.\n"
        "Return them as a simple numbered list, one per line."
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
        if line and line[0].isdigit():
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


def _kind_badge_label(kind: str) -> str:
    mapping = {
        "tagline": "Tagline",
        "header": "Hero header",
        "subheader": "Subheader",
        "body": "Body copy",
    }
    return mapping.get(kind, kind)


def _render_text_content(asset: TextAsset) -> None:
    """Render copy with a bit of visual hierarchy by kind."""
    txt = asset.content.strip()
    if asset.kind == "tagline":
        st.markdown(f"<div style='font-size:1.1rem; font-weight:600;'>{txt}</div>", unsafe_allow_html=True)
    elif asset.kind == "header":
        st.markdown(f"<div style='font-size:1.05rem; font-weight:600;'>{txt}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='font-size:0.95rem; line-height:1.4;'>{txt}</div>", unsafe_allow_html=True)


def main() -> None:
    # --- Global style tweaks (fonts, spacing, cards) ------------------------
    st.markdown(
        """
        <style>
        /* Slightly larger base font inside the main area */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .block-container p, .block-container li {
            font-size: 0.95rem;
        }
        /* Card-like container */
        .text-card {
            border-radius: 10px;
            padding: 0.85rem 0.9rem 0.8rem 0.9rem;
            border: 1px solid #e4e4e4;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
            margin-bottom: 0.75rem;
        }
        .text-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.35rem;
        }
        .text-card-meta {
            font-size: 0.78rem;
            color: #6b7280;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 500;
            background: #eef2ff;
            color: #4338ca;
        }
        .pill span {
            margin-right: 0.25rem;
        }
        .pill.favorite {
            background: #fef3c7;
            color: #92400e;
        }
        .panel {
            border-radius: 10px;
            border: 1px solid #e5e7eb;
            background: #f9fafb;
            padding: 0.9rem 1rem 1.1rem 1rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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
    st.markdown("### ✏️ Generate new copy")

    with st.container():
        st.markdown("<div class='panel'>", unsafe_allow_html=True)

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
                placeholder="e.g., funnier, more premium, more urgent…",
            )

        generate_col, _ = st.columns([1, 5])
        with generate_col:
            if st.button("Generate copy", type="primary", use_container_width=True):
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

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 📚 Existing copy for this campaign")

    # --- Existing text assets -----------------------------------------------
    if not campaign.text_assets:
        st.write("No copy generated yet. Use the controls above to generate some.")
        return

    filter_cols = st.columns([2, 2, 3])
    with filter_cols[0]:
        kind_filter = st.selectbox(
            "Filter by type",
            options=["all", "tagline", "header", "subheader", "body"],
            index=0,
        )
    with filter_cols[1]:
        show_only_favorites = st.checkbox("Show only favorites", value=False, help="Show only ★ items")

    filtered = campaign.text_assets
    if kind_filter != "all":
        filtered = [t for t in filtered if t.kind == kind_filter]
    if show_only_favorites:
        filtered = [t for t in filtered if t.is_favorite]

    if not filtered:
        st.write("No copy matches this filter yet.")
        return

    # Render cards in 2-column grid to reduce vertical scrolling
    grid_cols = st.columns(2)

    for idx, asset in enumerate(filtered):
        col = grid_cols[idx % 2]
        with col:
            with st.container():
                st.markdown("<div class='text-card'>", unsafe_allow_html=True)

                # Header row: badge + timestamp + favorite state
                st.markdown("<div class='text-card-header'>", unsafe_allow_html=True)

                left, right = st.columns([3, 1])

                with left:
                    badge_classes = "pill favorite" if asset.is_favorite else "pill"
                    st.markdown(
                        f"<span class='{badge_classes}'><span>●</span>{_kind_badge_label(asset.kind)}</span>",
                        unsafe_allow_html=True,
                    )
                    meta = asset.created_at.strftime("%Y-%m-%d %H:%M")
                    st.markdown(
                        f"<div class='text-card-meta'>Created {meta}</div>",
                        unsafe_allow_html=True,
                    )

                with right:
                    col_fav, col_del = st.columns([1, 1])

                    # Favorite toggle
                    with col_fav:
                        fav_label = "★" if asset.is_favorite else "☆"
                        if st.button(
                            fav_label,
                            key=f"fav_{asset.id}",
                            help="Toggle favorite",
                        ):
                            asset.is_favorite = not asset.is_favorite
                            save_campaign(campaign)
                            st.rerun()

                    # Delete icon
                    with col_del:
                        if st.button(
                            "🗑",
                            key=f"del_{asset.id}",
                            help="Delete this copy",
                        ):
                            # Remove asset from campaign
                            campaign.text_assets = [
                                t for t in campaign.text_assets if t.id != asset.id
                            ]
                            save_campaign(campaign)
                            st.rerun()


                st.markdown("</div>", unsafe_allow_html=True)  # close header

                # Copy itself
                _render_text_content(asset)

                st.markdown("</div>", unsafe_allow_html=True)  # close card


if __name__ == "__main__":
    main()
