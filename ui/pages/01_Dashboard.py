# ui/pages/01_Dashboard.py
from __future__ import annotations

from typing import List

import shutil
import streamlit as st

import caf_app.campaign_generator as campaign_generator
from caf_app.storage import campaigns_dir, load_campaign, save_campaign


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


def _delete_campaign(slug: str) -> None:
    """
    Delete the entire campaign folder for the given slug.
    """
    root = campaigns_dir()
    target = root / slug
    if target.exists() and target.is_dir():
        shutil.rmtree(target)


def _rename_campaign(slug: str, new_name: str) -> None:
    """
    Rename a campaign by updating its stored name (NOT the slug/folder).
    Safer: we keep the same slug and just change the human-readable name.
    """
    campaign = load_campaign(slug)
    if campaign is None:
        raise RuntimeError(f"Could not load campaign '{slug}' for rename.")

    # Try common name attributes
    if hasattr(campaign, "name"):
        campaign.name = new_name
    if hasattr(campaign, "campaign_name"):
        setattr(campaign, "campaign_name", new_name)

    save_campaign(campaign)


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("Dashboard – Campaigns")

    # Make sure campaigns root exists
    campaigns_root = campaigns_dir()
    campaigns_root.mkdir(parents=True, exist_ok=True)

    # Try to get currently selected campaign (if any)
    current_slug = st.session_state.get("current_campaign_slug")

    col_left, col_right = st.columns([2, 1])

    # ----------------------------------------------------------------------
    # LEFT: Create / regenerate a campaign (TEXT ONLY)
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

            submitted = st.form_submit_button("Generate campaign", type="primary")

        if submitted:
            if not campaign_name.strip():
                st.error("Please enter a campaign name.")
            elif not campaign_brief.strip():
                st.error("Please enter a campaign brief.")
            else:
                with st.status("Generating campaign (text only)...", expanded=True) as status:
                    try:
                        st.write("• Generating text assets...")
                        result = campaign_generator.generate_campaign(
                            campaign_name=campaign_name,
                            campaign_brief=campaign_brief,
                        )

                        slug = result["slug"]
                        _set_current_campaign(slug)

                        st.write("• Text assets generated.")
                        st.write(f"• Campaign folder: `{result['campaign_dir']}`")
                        st.write(f"• ZIP export: `{result['zip_path']}`")
                        status.update(
                            label="Campaign generated successfully (text only).",
                            state="complete",
                        )

                        st.success(
                            f"✅ Campaign **{campaign_name}** created with slug `{slug}`.\n\n"
                            "Use the sidebar to open the Brief, Text Library, and Image Library "
                            "to generate images later."
                        )
                    except Exception as e:
                        status.update(label="Campaign generation failed.", state="error")
                        st.error(f"Error while generating campaign: {e}")
                        st.exception(e)

    # ----------------------------------------------------------------------
    # RIGHT: Existing campaigns (select / open / rename / delete)
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
                "Select a campaign",
                options=slugs,
                index=default_index,
                format_func=lambda s: s,
                help="Choose an existing campaign to continue working on it.",
            )

            if st.button("Open campaign"):
                _set_current_campaign(selected_slug)
                st.success(
                    f"Opened campaign `{selected_slug}`. "
                    "Use the sidebar to navigate to the Brief, Text Library, or Image Library."
                )

            # Show a small preview (name + brief snippet) for context
            name = selected_slug
            brief = ""
            try:
                if selected_slug:
                    campaign = load_campaign(selected_slug)
                    if campaign is not None:
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

            st.markdown("---")
            st.subheader("Manage campaign")

            # Rename campaign (name only, slug stays the same)
            new_name = st.text_input(
                "Rename campaign (changes display name only)",
                value=name,
                key=f"rename_{selected_slug}",
            )

            col_rename, col_delete = st.columns(2)

            with col_rename:
                if st.button("Save new name", key=f"btn_rename_{selected_slug}"):
                    if not new_name.strip():
                        st.error("New name cannot be empty.")
                    else:
                        try:
                            _rename_campaign(selected_slug, new_name.strip())
                            st.success(f"Renamed campaign `{selected_slug}` to “{new_name.strip()}”.")
                        except Exception as e:
                            st.error(f"Failed to rename campaign `{selected_slug}`: {e}")
                            st.exception(e)

            with col_delete:
                if st.button("Delete campaign", key=f"btn_delete_{selected_slug}"):
                    try:
                        _delete_campaign(selected_slug)
                        if current_slug == selected_slug:
                            st.session_state.pop("current_campaign_slug", None)
                        st.success(f"Deleted campaign `{selected_slug}`.")
                        # Force a rerun so the selectbox updates
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete campaign `{selected_slug}`: {e}")
                        st.exception(e)


if __name__ == "__main__":
    main()
