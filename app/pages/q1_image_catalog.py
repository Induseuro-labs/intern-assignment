"""
Question 1 Streamlit Entry Point.
Redirects to the tactile Skeuomorphic and Claymorphic Engineering Console.
"""

import os
import streamlit as st
import streamlit.components.v1 as components

FAVICON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "favicon.svg")

st.set_page_config(
    page_title="Question 1 - Image Catalog Console",
    page_icon=FAVICON_PATH if os.path.exists(FAVICON_PATH) else None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
  #MainMenu, header, footer, [data-testid="stSidebar"], [data-testid="stToolbar"], [data-testid="stHeader"] {
    display: none !important;
    visibility: hidden !important;
  }
  .main .block-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
  }
  .stApp {
    background-color: #e5ebf3 !important;
  }
  iframe {
    width: 100vw !important;
    height: 100vh !important;
    border: none !important;
    display: block !important;
  }
</style>
""", unsafe_allow_html=True)

components.iframe(src="http://127.0.0.1:8000", height=960, scrolling=True)

def render():
    pass
