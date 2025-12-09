# ui/home.py

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# -------------------------------------------------------------------
# Page config (must be first Streamlit call)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Content Asset Factory – MVP 1",
    layout="wide",
)

# -------------------------------------------------------------------
# Global layout / styling tweaks
# -------------------------------------------------------------------
# Make the main content container wider and left-aligned rather than centered
st.markdown(
    """
    <style>
        /* Main block container (the core page content area) */
        .block-container {
            max-width: 1400px;      /* wider than default */
            padding-left: 2rem;
            padding-right: 2rem;
            margin-left: 0;         /* align to left instead of centered */
        }

        /* Extra safety: ensure the outer app containers aren't forcing centering */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
            margin-left: 0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Home page content
# -------------------------------------------------------------------
st.title("Content Asset Factory – MVP 1")

st.write(
    "Use the sidebar to navigate:\n\n"
    "1. **Dashboard** – create or select a campaign.\n"
    "2. **Brief Editor** – define the campaign brief.\n"
    "3. **Text Library** – generate and curate copy.\n"
    "4. **Image Library** – generate and curate images.\n"
    "5. **Concept Builder** – mix text + images into concepts.\n"
    "6. **Review & Export** – pick final concepts and export."
)
