# ui/02_Brief_Editor.py
from __future__ import annotations

import streamlit as st

from caf_app.storage import load_campaign, save_campaign


def _get_current_slug() -> str | None:
    return st.session_state.get("current_campaign_slug")


def main() -> None:
    # --- Light styling to match upgraded Text Library -----------------------
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .block-container p, .block-container li {
            font-size: 0.95rem;
        }
        .brief-panel {
            border-radius: 10px;
            border: 1px solid #e5e7eb;
            background: #f9fafb;
            padding: 1rem 1.1rem 1.2rem 1.1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }
        .brief-tip {
            border-radius: 10px;
            border: 1px dashed #e5e7eb;
            background: #fefce8;
            padding: 0.75rem 0.9rem;
            font-size: 0.9rem;
        }
        .brief-tip-title {
            font-weight: 600;
            margin-bottom: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Campaign Brief Editor")

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

    st.caption(f"Editing campaign: **{campaign.name}** (`{campaign.slug}`)")

    st.markdown("### 📝 Brief details")

    # Wrap the form in a subtle card/panel
    st.markdown("<div class='brief-panel'>", unsafe_allow_html=True)

    with st.form("brief_form"):
        # Product + tone on one row to use space better
        col1, col2 = st.columns([2, 1])

        with col1:
            product_name = st.text_input(
                "Product name",
                value=campaign.product_name or "",
                placeholder="e.g., Nano Banana Energy Drink",
            )
        with col2:
            tone = st.text_input(
                "Tone / style",
                value=campaign.tone or "",
                placeholder="e.g., playful, premium, bold, minimalist…",
            )

        description = st.text_area(
            "Campaign description",
            value=campaign.description or "",
            placeholder=(
                "What is this campaign about? Key message, value proposition, goals..."
            ),
            height=110,
        )

        col3, col4 = st.columns(2)
        with col3:
            audience = st.text_area(
                "Target audience",
                value=campaign.audience or "",
                placeholder="Who are you trying to reach? Demographics, interests, etc.",
                height=90,
            )
        with col4:
            notes = st.text_area(
                "Additional notes / constraints",
                value=campaign.notes or "",
                placeholder="Brand rules, mandatory phrases, do/don'ts, etc.",
                height=90,
            )

        submit_col, _ = st.columns([1, 5])
        with submit_col:
            submitted = st.form_submit_button("Save Brief", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        campaign.product_name = product_name.strip() or None
        campaign.description = description.strip() or None
        campaign.audience = audience.strip() or None
        campaign.tone = tone.strip() or None
        campaign.notes = notes.strip() or None

        save_campaign(campaign)

        st.success("Brief saved successfully.")
        st.info("You can now go to **Text Library** or **Image Library** to generate assets.")

    st.markdown("---")

    # Tip box with a bit more presence
    st.markdown(
        """
        <div class="brief-tip">
          <div class="brief-tip-title">Tip</div>
          <div>
            Keep the brief concise but concrete. A clear product, audience, and tone
            will make both text and image generation much stronger.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
