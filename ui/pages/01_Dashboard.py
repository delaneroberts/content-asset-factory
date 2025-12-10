# ui/pages/01_Dashboard.py
from __future__ import annotations

from pathlib import Path
from typing import List

import streamlit as st

from caf_app.campaign_generator import generate_campaign
from caf_app.storage import campaigns_dir, load_campaign


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_campaign_slugs() -> List[str]:
    """
    Return a list of campaign slugs based on folders under campaigns/.
    """
    root = campaigns_dir()
    if not root.exists():
        return []
    return sorted(
        [p.name for p in root.iterdir() if p.is_dir()],
        key=str.lower,
    )


def _set_current_campaign(slug: str) -> None:
    st.session_state["current_campaign_slug"] = slug


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main():
    st.title("Dashboard – Campaigns")

    # Make sure campaigns root exists
    campaigns_root = campaigns_dir()
    campaigns_root.mkdir(parents=True, exist_ok=True)

    # Current selection from session
    current_slug = st.session_state.get("current_campaign_slug")

    # ----------------------------------------------------------------------
    # Layout: left = create, right = existing
    # ----------------------------------------------------------------------
    col_left, col_right = st.columns([2, 1])

    # ----------------------------------------------------------------------
    # LEFT: Create / regenerate a campaign
    # ----------------------------------------------------------------------
    with col_left:
        st.subheader("Create a new campaign")

        with st.form("create_campaign_form"):
            campaign_name = st.text_input("Campaign name", value="")
            campaign_brief = st.text_area(
                "Campaign brief",
                value="",
                height=160,
                help="Describe the product, audience, tone, and goals.",
            )

            image_engine = st.selectbox(
                "Image engine",
                options=["auto", "openai", "nanobanana", "stability"],
                index=0,
                help=(
                    "auto: try OpenAI → NanoBanana → Stability.\n"
                    "openai: OpenAI only.\n"
                    "nanobanana: NanoBanana only.\n"
                    "stability: Stability SDXL only."
                ),
)


            num_supporting = st.slider(
                "Number of supporting images",
                min_value=1,
                max_value=6,
                value=3,
                help="How many supporting images to generate alongside the hero.",
            )

            submitted = st.form_submit_button("Generate campaign", type="primary")

        if submitted:
            if not campaign_name.strip():
                st.error("Please enter a campaign name.")
            elif not campaign_brief.strip():
                st.error("Please enter a campaign brief.")
            else:
                with st.status("Generating campaign ...", expanded=True) as status:
                    try:
                        st.write("• Generating text assets...")
                        result = generate_campaign(
                            campaign_name=campaign_name,
                            campaign_brief=campaign_brief,
                            image_engine=image_engine,
                            num_supporting=num_supporting,
                        )

                        slug = result["slug"]
                        _set_current_campaign(slug)

                        st.write("• Text and images generated.")
                        st.write(f"• Campaign folder: `{result['campaign_dir']}`")
                        st.write(f"• ZIP export: `{result['zip_path']}`")
                        status.update(label="Campaign generated successfully.", state="complete")

                        st.success(
                            f"✅ Campaign **{campaign_name}** created with slug `{slug}`.\n\n"
                            "Use the sidebar to open the Brief, Text Library, and Image Library."
                        )
                    except Exception as e:
                        status.update(label="Campaign generation failed.", state="error")
                        st.error(f"Error while generating campaign: {e}")
                        st.exception(e)

    # ----------------------------------------------------------------------
    # RIGHT: Existing campaigns
    # ----------------------------------------------------------------------
    with col_right:
        st.subheader("Existing campaigns")

        slugs = _list_campaign_slugs()
        if not slugs:
            st.info("No campaigns yet. Create one on the left.")
        else:
            # Try to default to current slug if present
            if current_slug in slugs:
                default_index = slugs.index(current_slug)
            else:
                default_index = 0

            selected_slug = st.selectbox(
                "Select campaign",
                slugs,
                index=default_index,
                key="campaign_select_box",
            )

            if st.button("Load selected campaign"):
                _set_current_campaign(selected_slug)
                st.success(f"Loaded campaign `{selected_slug}`.")

            # Show a tiny summary
            try:
                if selected_slug:
                    campaign = load_campaign(selected_slug)
                    name = getattr(campaign, "name", selected_slug)
                    brief = getattr(campaign, "campaign_brief", "") or getattr(
                        campaign, "brief", ""
                    )
                    st.markdown("**Name:** " + name)
                    if brief:
                        st.markdown("**Brief (preview):**")
                        st.caption(brief[:200] + ("..." if len(brief) > 200 else ""))
            except Exception as e:
                st.error(f"Failed to load campaign `{selected_slug}`: {e}")


if __name__ == "__main__":
    main()
