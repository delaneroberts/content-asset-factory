from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import streamlit as st

from caf_app.campaign_generator import generate_campaign
from caf_app.image_gen import ImageProvider


# ---------- Page config ----------

st.set_page_config(
    page_title="Content Asset Factory",
    page_icon="🎨",
    layout="wide",
)


# ---------- Global Styles ----------

st.markdown(
    """
    <style>
    /* App background gradient */
    .stApp {
        background: radial-gradient(circle at top left, #4f46e5 0, #2563eb 30%, #06b6d4 60%, #22c55e 100%);
    }

    /* Main content container */
    .block-container {
        max-width: 1180px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    /* Hero banner at top */
    .caf-hero-banner {
        background: linear-gradient(90deg, rgba(15,23,42,0.95), #1d4ed8, #0ea5e9);
        border-radius: 24px;
        padding: 18px 22px;
        margin-bottom: 20px;
        color: white;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .caf-hero-icon {
        font-size: 2.2rem;
    }
    .caf-hero-title {
        font-size: 1.5rem;
        font-weight: 650;
        margin: 0;
    }
    .caf-hero-subtitle {
        font-size: 0.95rem;
        margin: 2px 0 0 0;
        opacity: 0.9;
    }

    /* Generic card styling */
    .caf-card {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 18px 20px 16px 20px;
        border: 1px solid rgba(148,163,184,0.35);
        box-shadow: 0 10px 25px rgba(15,23,42,0.16);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        margin-bottom: 14px;
    }

    /* Section titles inside cards */
    .caf-section-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    /* Small pill labels */
    .caf-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        border: 1px solid #e5e7eb;
        background: #f9fafb;
        color: #4b5563;
        margin-right: 6px;
        margin-bottom: 4px;
    }

    /* Success badge */
    .caf-success-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        background: #ecfdf5;
        color: #15803d;
        border: 1px solid #bbf7d0;
        margin-bottom: 10px;
    }

    /* Streamlit button tweak */
    .stButton > button {
        width: 100%;
        border-radius: 999px;
        height: 42px;
        border: none;
        font-weight: 600;
        background: linear-gradient(90deg, #2563eb, #4f46e5);
        color: white;
        box-shadow: 0 8px 18px rgba(37,99,235,0.45);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #4338ca);
        box-shadow: 0 10px 24px rgba(30,64,175,0.55);
    }

    /* Tabs tighter spacing */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 6px;
        padding-bottom: 6px;
        padding-left: 10px;
        padding-right: 10px;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Hero banner ----------

st.markdown(
    """
    <div class="caf-hero-banner">
      <div class="caf-hero-icon">🎨</div>
      <div>
        <p class="caf-hero-title">Content Asset Factory</p>
        <p class="caf-hero-subtitle">
          Generate a ready-to-hand-off campaign package – hero & supporting images,
          copy, and a downloadable ZIP for designers or clients.
        </p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Main layout: three columns (inputs | generated | right rail) ----------

col_inputs, col_generated, col_right = st.columns([1.1, 1.35, 1.0])

# ---------- Left: Campaign inputs (inside a form) ----------

with col_inputs:
    with st.form("campaign_form"):
        st.markdown('<div class="caf-card">', unsafe_allow_html=True)
        st.markdown('<div class="caf-section-title">Campaign inputs</div>', unsafe_allow_html=True)

        campaign_name = st.text_input(
            "Campaign name",
            value="Evo Soda Launch",
            placeholder="e.g., Evo Soda Summer Launch",
        )

        campaign_brief = st.text_area(
            "Campaign brief",
            height=190,
            placeholder=(
                "Describe the product, audience, tone, and goals.\n\n"
                "Example: A new zero-sugar sparkling water aimed at busy "
                "professionals who want something refreshing but healthy..."
            ),
        )

        st.markdown("")  # spacing
        submitted = st.form_submit_button("Generate campaign 🚀🎨")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Right rail: Settings card (always visible) ----------

with col_right:
    st.markdown('<div class="caf-card">', unsafe_allow_html=True)
    st.markdown('<div class="caf-section-title">Settings</div>', unsafe_allow_html=True)

    provider_label = st.selectbox(
        "Image provider",
        [
            "Placeholder only",
            "OpenAI (gpt-image-1)",
            "Gemini (image-1)",
        ],
        index=0,
        help=(
            "For this demo, all providers currently use placeholder images, "
            "but the selected value is stored in campaign metadata."
        ),
    )

    provider_map: Dict[str, ImageProvider] = {
        "OpenAI (gpt-image-1)": ImageProvider.OPENAI,
        "Gemini (image-1)": ImageProvider.GEMINI,
        "Placeholder only": ImageProvider.PLACEHOLDER,
    }

    st.caption(
        "Generated assets are saved under the `campaigns/<slug>/` folder "
        "in your project."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------- Handle submission & show results ----------

if submitted:
    if not campaign_name.strip() or not campaign_brief.strip():
        with col_generated:
            st.error("Please provide both a **campaign name** and a **brief**.")
    else:
        image_provider: ImageProvider = provider_map[provider_label]

        with col_generated:
            with st.spinner(f"Generating campaign using {provider_label}..."):
                result: Dict[str, Any] = generate_campaign(
                    campaign_name=campaign_name,
                    campaign_brief=campaign_brief,
                    image_provider=image_provider,
                )

        # Unpack result
        slug = result["slug"]
        campaign_dir: Path = result["campaign_dir"]
        text_assets = result["text_assets"]
        hero_path: Path = result["hero_path"]
        supporting_paths = result["supporting_paths"]
        zip_path: Path = result["zip_path"]

        # ----- Center column: Generated campaign card -----

        with col_generated:
            st.markdown('<div class="caf-card">', unsafe_allow_html=True)
            st.markdown('<div class="caf-section-title">Generated campaign</div>', unsafe_allow_html=True)

            st.markdown(
                '<div class="caf-success-pill">✅ Campaign generated successfully</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="margin-bottom: 6px;">
                  <span class="caf-pill">Slug: <code>{slug}</code></span>
                  <span class="caf-pill">Provider: <code>{image_provider.value}</code></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Images + key text in a sub layout
            img_col, text_col = st.columns([1.05, 1.0])

            with img_col:
                st.image(str(hero_path), caption="Hero image", width="stretch")

                st.markdown('<div class="caf-section-title" style="margin-top:0.6rem;">Supporting images</div>', unsafe_allow_html=True)
                support_cols = st.columns(3)
                for i, (p, c) in enumerate(zip(supporting_paths, support_cols), start=1):
                    with c:
                        st.image(str(p), caption=f"{i}", width="stretch")

            with text_col:
                st.markdown('<div class="caf-section-title">Tagline</div>', unsafe_allow_html=True)
                st.write(text_assets["tagline"])

                st.markdown('<div class="caf-section-title" style="margin-top:0.7rem;">Value proposition</div>', unsafe_allow_html=True)
                st.write(text_assets["value_prop"])

                st.markdown('<div class="caf-section-title" style="margin-top:0.7rem;">Call-to-actions</div>', unsafe_allow_html=True)
                for line in text_assets["ctas"].splitlines():
                    line = line.strip()
                    if line:
                        st.markdown(f"- **{line}**")

            st.markdown("</div>", unsafe_allow_html=True)

        # ----- Right rail: Key copy card -----

        with col_right:
            st.markdown('<div class="caf-card">', unsafe_allow_html=True)
            st.markdown('<div class="caf-section-title">Key copy</div>', unsafe_allow_html=True)

            st.markdown("**Tagline**")
            st.write(text_assets["tagline"])

            st.markdown("**Slogans**")
            for line in text_assets["slogans"].splitlines():
                line = line.strip()
                if line:
                    st.markdown(f"- {line}")

            st.markdown("**Call-to-actions**")
            for line in text_assets["ctas"].splitlines():
                line = line.strip()
                if line:
                    st.markdown(f"- {line}")

            st.markdown("</div>", unsafe_allow_html=True)

            # Files card
            st.markdown('<div class="caf-card">', unsafe_allow_html=True)
            st.markdown('<div class="caf-section-title">Files & ZIP</div>', unsafe_allow_html=True)
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
    # Nothing submitted yet – show a subtle hint in center column
    with col_generated:
        st.markdown('<div class="caf-card">', unsafe_allow_html=True)
        st.markdown('<div class="caf-section-title">Generated campaign</div>', unsafe_allow_html=True)
        st.info(
            "When you generate a campaign, the hero image, supporting images, and key copy "
            "will appear here."
        )
        st.markdown("</div>", unsafe_allow_html=True)