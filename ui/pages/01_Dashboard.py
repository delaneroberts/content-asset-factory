# ui/01_Dashboard.py
from __future__ import annotations

import streamlit as st

from caf_app.storage import create_campaign, list_campaigns, delete_campaign


def _init_session_state() -> None:
    if "current_campaign_slug" not in st.session_state:
        st.session_state.current_campaign_slug = None
    if "confirm_delete_slug" not in st.session_state:
        st.session_state.confirm_delete_slug = None


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

    confirm_slug = st.session_state.get("confirm_delete_slug")

    for campaign in campaigns:
        is_current = (
            st.session_state.current_campaign_slug == campaign.slug
        )

        # Added an extra column for the trash icon
        cols = st.columns([4, 3, 3, 2, 1])

        # --- Campaign name + slug ---
        with cols[0]:
            st.markdown(f"**{campaign.name}**")
            slug_display = f"`{campaign.slug}`"

            if is_current:
                st.caption(slug_display + "  \n✅ *Current campaign*")
            else:
                st.caption(slug_display)

        # --- Created timestamp ---
        with cols[1]:
            st.caption(f"Created: {campaign.created_at.strftime('%Y-%m-%d %H:%M')}")

        # --- Updated timestamp ---
        with cols[2]:
            st.caption(f"Updated: {campaign.updated_at.strftime('%Y-%m-%d %H:%M')}")

        # --- Open / Selected button ---
        with cols[3]:
            button_label = "Selected" if is_current else "Open"

            if st.button(button_label, key=f"open_{campaign.slug}"):
                # Only switch if clicking on a NEW selection
                if not is_current:
                    st.session_state.current_campaign_slug = campaign.slug
                    st.success(
                        f"Selected **{campaign.name}**. "
                        "Go to **Brief Editor** to view or edit its brief."
                    )
                    st.rerun()

        # --- Delete (trashcan icon) ---
        with cols[4]:
            if st.button("🗑", key=f"del_{campaign.slug}", help="Delete this campaign"):
                st.session_state.confirm_delete_slug = campaign.slug
                st.rerun()

    # --- Delete confirmation section ---
    if confirm_slug:
        st.warning(f"Are you sure you want to delete campaign `{confirm_slug}`?")
        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("Yes, delete permanently"):
                delete_campaign(confirm_slug)
                # Clear current selection if we just deleted it
                if st.session_state.current_campaign_slug == confirm_slug:
                    st.session_state.current_campaign_slug = None
                st.session_state.confirm_delete_slug = None
                st.success("Campaign deleted.")
                st.rerun()

        with col_b:
            if st.button("Cancel"):
                st.session_state.confirm_delete_slug = None
                st.rerun()


if __name__ == "__main__":
    main()
