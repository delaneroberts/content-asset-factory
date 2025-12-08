# ui/02_Brief_Editor.py
from __future__ import annotations

import streamlit as st

from caf_app.storage import load_campaign, save_campaign


def _get_current_slug() -> str | None:
    return st.session_state.get("current_campaign_slug")


def main() -> None:
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

    st.markdown("### Brief details")

    with st.form("brief_form"):
        product_name = st.text_input(
            "Product name",
            value=campaign.product_name or "",
            placeholder="e.g., Nano Banana Energy Drink",
        )

        description = st.text_area(
            "Campaign description",
            value=campaign.description or "",
            placeholder=(
                "What is this campaign about? Key message, value proposition, goals..."
            ),
            height=120,
        )

        audience = st.text_area(
            "Target audience",
            value=campaign.audience or "",
            placeholder="Who are you trying to reach? Demographics, interests, etc.",
            height=80,
        )

        tone = st.text_input(
            "Tone / style",
            value=campaign.tone or "",
            placeholder="e.g., playful, premium, bold, minimalist...",
        )

        notes = st.text_area(
            "Additional notes / constraints",
            value=campaign.notes or "",
            placeholder="Brand rules, mandatory phrases, do/don'ts, etc.",
            height=100,
        )

        submitted = st.form_submit_button("Save Brief")

    if submitted:
        campaign.product_name = product_name.strip() or None
        campaign.description = description.strip() or None
        campaign.audience = audience.strip() or None
        campaign.tone = tone.strip() or None
        campaign.notes = notes.strip() or None

        # updated_at will be refreshed inside save_campaign (or you can set it here)
        save_campaign(campaign)

        st.success("Brief saved successfully.")
        st.info("You can now go to **Text Library** or **Image Library** to generate assets.")

    st.markdown("---")
    st.markdown(
        "#### Tip\n"
        "Keep the brief concise but concrete. The same brief will drive both text and image generation."
    )


if __name__ == "__main__":
    main()

