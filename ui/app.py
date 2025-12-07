from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import streamlit as st

from caf_app.campaign_generator import generate_campaign
from caf_app.image_gen import ImageProvider


# ---------- Page config ----------

st.set_page_config(
    page_title="Content Asset Factory",
    page_icon="📣",
    layout="wide",
)


# ---------- Global Styles (simple CSS) ----------

st.markdown(
    """
    <style>
    /* Overall page tweaks */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Top banner */
    .caf-hero-banner {
        background: linear-gradient(90deg, #111827, #1f2937, #4b5563);
        border-radius: 18px;
        padding: 18px 22px;
        margin-bottom: 18px;
        color: white;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .caf-hero-icon {
        font-size: 1.9rem;
    }
    .caf-hero-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0;
    }
    .caf-hero-subtitle {
        font-size: 0.9rem;
        margin: 2px 0 0 0;
        opacity: 0.85;
    }

    /* Card styling */
    .caf-card {
        background-color: #ffffff;
        border-radius: 14px;
        padding: 18px 18px 14px 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
        margin-bottom: 14px;
    }

    /* Section headers */
    .caf-section-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    /* Small tag/pill */
    .caf-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 2px 9px;
        border-radius: 999px;
        font-size: 0.75rem;
        border: 1px solid #e5e7eb;
        background: #f9fafb;
        color: #4b5563;
        margin-right: 6px;
        margin-bottom: 4px;
    }

    /* Tabs: slightly tighter */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 6px;
        padding-bottom: 6px;
        padding-left: 10px;
        padding-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Sidebar: settings ----------

st.sidebar.title("⚙️ Settings")

provider_label = st.sidebar.selectbox(
    "Image provider",
    [
        "Placeholder only",
        "OpenAI (gpt-image-1)",
        "Gemini (image-1)",
    ],
    index=0,
    help=(
        "For this demo, all providers currently use placeholder images, "
        "but the selected provider is recorded in campaign metadata."
    ),
)

provider_map: Dict[str, ImageProvider] = {
    "OpenAI (gpt-image-1)": ImageProvider.OPENAI,
    "Gemini (image-1)": ImageProvider.GEMINI,
    "Placeholder only": ImageProvider.PLACEHOLDER,
}

st.sidebar.markdown("---")
st.sidebar.caption(
    "Generated assets are saved under the `campaigns/` folder in your project."
)


# ---------- Top banner ----------

st.markdown(
    """
    <div class="caf-hero-banner">
      <div class="caf-hero-icon">📣</div>
      <div>
        <p class="caf-hero-title">Content Asset Factory</p>
        <p class="caf-hero-subtitle">
          Generate a ready-to-hand-off campaign package: hero & supporting images,
          copy, and a downloadable ZIP for designers or clients.
        </p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Main form ----------

with st.container():
    with st.form("campaign_form"):
        st.markdown("#### Campaign inputs")

        col_name, col_brief = st.columns([1, 2])

        with col_name:
            campaign_name = st.text_input(
                "Campaign name",
                value="Evo Soda Launch",
                placeholder="e.g., Evo Soda Summer Launch",
            )

        with col_brief:
            campaign_brief = st.text_area(
                "Campaign brief",
                height=180,
                placeholder=(
                    "Describe the product, audience, tone, and goals.\n\n"
                    "Example: A new zero-sugar sparkling water aimed at busy "
                    "professionals who want something refreshing but healthy..."
                ),
            )

        submitted = st.form_submit_button("Generate campaign 🚀")


# ---------- Handle submission ----------

if submitted:
    if not campaign_name.strip() or not campaign_brief.strip():
        st.error("Please provide both a **campaign name** and a **brief**.")
    else:
        image_provider: ImageProvider = provider_map[provider_label]

        with st.spinner(f"Generating campaign using {provider_label}..."):
            result: Dict[str, Any] = generate_campaign(
                campaign_name=campaign_name,
                campaign_brief=campaign_brief,
                image_provider=image_provider,
            )

        st.success("Campaign generated successfully ✅")

        slug = result["slug"]
        campaign_dir: Path = result["campaign_dir"]
        text_assets = result["text_assets"]
        hero_path: Path = result["hero_path"]
        supporting_paths = result["supporting_paths"]
        zip_path: Path = result["zip_path"]

        # ---------- Summary / metadata card ----------

        with st.container():
            st.markdown(
                f"""
                <div class="caf-card">
                  <div class="caf-section-title">Campaign overview</div>
                  <div style="margin-bottom: 4px;">
                    <span class="caf-pill">Slug: <code>{slug}</code></span>
                    <span class="caf-pill">Provider: <code>{image_provider.value}</code></span>
                  </div>
                  <div style="font-size:0.85rem; color:#4b5563; margin-top:4px;">
                    Folder: <code>campaigns/{slug}</code><br/>
                    ZIP: <code>{zip_path.name}</code>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------- Layout: images + key copy ----------

        col_left, col_right = st.columns([1.4, 1])

        # Left: images
        with col_left:
            st.markdown('<div class="caf-card">', unsafe_allow_html=True)
            st.markdown('<div class="caf-section-title">Hero image</div>', unsafe_allow_html=True)

            st.image(str(hero_path), caption="Hero", width="stretch")

            st.markdown('<div class="caf-section-title" style="margin-top:0.6rem;">Supporting images</div>', unsafe_allow_html=True)

            cols_support = st.columns(3)
            for i, (p, c) in enumerate(zip(supporting_paths, cols_support), start=1):
                with c:
                    st.image(str(p), caption=f"Support {i}", width="stretch")

            st.markdown("</div>", unsafe_allow_html=True)

        # Right: core copy
        with col_right:
            st.markdown('<div class="caf-card">', unsafe_allow_html=True)

            st.markdown('<div class="caf-section-title">Tagline</div>', unsafe_allow_html=True)
            st.write(text_assets["tagline"])

            st.markdown('<div class="caf-section-title" style="margin-top:0.8rem;">Slogans</div>', unsafe_allow_html=True)
            for line in text_assets["slogans"].splitlines():
                line = line.strip()
                if line:
                    st.markdown(f"- {line}")

            st.markdown('<div class="caf-section-title" style="margin-top:0.8rem;">Value proposition</div>', unsafe_allow_html=True)
            st.write(text_assets["value_prop"])

            st.markdown('<div class="caf-section-title" style="margin-top:0.8rem;">Call-to-actions</div>', unsafe_allow_html=True)
            for line in text_assets["ctas"].splitlines():
                line = line.strip()
                if line:
                    st.markdown(f"- **{line}**")

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Tabs for details ----------

        st.markdown("")  # small spacing
        with st.container():
            st.markdown('<div class="caf-card">', unsafe_allow_html=True)

            tab_social, tab_summary, tab_prompts, tab_files = st.tabs(
                ["📱 Social posts", "📝 Summary", "🧠 Prompts used", "📂 Files & ZIP"]
            )

            with tab_social:
                st.subheader("Social posts")
                st.code(text_assets["social_posts"], language="markdown")

            with tab_summary:
                st.subheader("Campaign summary")
                st.write(text_assets["summary"])

            with tab_prompts:
                st.subheader("LLM prompts used")
                st.caption(
                    "These are the exact prompts used to generate the copy, "
                    "saved as `generation_prompts.txt` inside the campaign folder."
                )
                st.code(text_assets["_prompts"], language="markdown")

            with tab_files:
                st.subheader("Files")
                st.markdown(f"- Folder: `{campaign_dir}`")
                st.markdown(f"- ZIP: `{zip_path.name}`")

                if zip_path.exists():
                    with zip_path.open("rb") as f:
                        zip_bytes = f.read()

                    st.download_button(
                        label="⬇️ Download campaign ZIP",
                        data=zip_bytes,
                        file_name=zip_path.name,
                        mime="application/zip",
                    )
                else:
                    st.warning("ZIP file not found. Please check the backend.")

            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Fill in the campaign name and brief, then click **Generate campaign 🚀**.")
