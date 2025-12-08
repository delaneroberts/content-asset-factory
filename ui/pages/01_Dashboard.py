# ui/01_Dashboard.py
from __future__ import annotations

import streamlit as st

from caf_app.storage import create_campaign, list_campaigns

current_slug = st.session_state.get("current_campaign_slug")

def _init_session_state() -> None:
    if "current_campaign_slug" not in st.session_state:
        st.session_state.current_campaign_slug = None


def main() -> None:
    _init_session_state()

    st.title("Campaign Dashboard")

    st.markdown("### Create a new campaign")

    with st.form("new_campaign_form", clear_on_submit=True):
        name = st.text_input("Campaign name", placeholder="Spring Launch 2025")
        submitted = st.form_submit_button("Create Campaign")

    if submitted:
        if not name.strip():
            st.warning("Please enter a campaign name.")
        else:
            campaign = create_campaign(name.strip())
            st.session_state.current_campaign_slug = campaign.slug
            st.success(f"Created campaign **{campaign.name}** (slug: `{campaign.slug}`).")
            st.info("Now switch to **Brief Editor** in the sidebar to define the brief.")

    st.markdown("---")
    st.markdown("### Existing campaigns")

    campaigns = list_campaigns()
    if not campaigns:
        st.write("No campaigns yet. Create one above.")
        return

    for campaign in campaigns:
        is_current = (
            st.session_state.current_campaign_slug == campaign.slug
        )

        cols = st.columns([4, 3, 3, 2])
        with cols[0]:
            st.markdown(f"**{campaign.name}**")
            st.caption(f"`{campaign.slug}`")
        with cols[1]:
            st.caption(f"Created: {campaign.created_at.strftime('%Y-%m-%d %H:%M')}")
        with cols[2]:
            st.caption(f"Updated: {campaign.updated_at.strftime('%Y-%m-%d %H:%M')}")
        with cols[3]:
            if st.button(
                "Open",
                key=f"open_{campaign.slug}",
            ):
                st.session_state.current_campaign_slug = campaign.slug
                st.success(
                    f"Selected **{campaign.name}**. "
                    "Go to **Brief Editor** to view or edit its brief."
                )

        if is_current:
            st.markdown(
                "<span style='color: green; font-weight: 600;'>Current campaign</span>",
                unsafe_allow_html=True,
            )

        st.markdown("---")


if __name__ == "__main__":
    main()
