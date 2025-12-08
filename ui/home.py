# ui/home.py
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="Content Asset Factory – MVP 1",
    layout="wide",
)

st.title("Content Asset Factory – MVP 1")

st.write(
    "Use the sidebar to navigate:\n\n"
    "1. Dashboard – create or select a campaign.\n"
    "2. Brief Editor – define the campaign brief.\n"
    "3. Text Library – generate and curate copy.\n"
    "4. Image Library – generate and curate images.\n"
    "5. Concept Builder – mix text + images into concepts.\n"
    "6. Review & Export – pick final concepts and export."
)
