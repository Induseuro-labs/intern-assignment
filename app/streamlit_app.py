"""
Main Streamlit Entry Point.
Serves the tactile Skeuomorphic and Claymorphic Engineering Console.
Hides Streamlit default chrome and seamlessly displays the full tactile console.
"""

from __future__ import annotations
import os
import socket
import threading
import streamlit as st
import streamlit.components.v1 as components

# Ensure FastAPI backend on port 8000 is active
def ensure_backend():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    res = sock.connect_ex(("127.0.0.1", 8000))
    sock.close()
    if res != 0:
        def run_server():
            import uvicorn
            from app.server import app
            uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")
        t = threading.Thread(target=run_server, daemon=True)
        t.start()

ensure_backend()

FAVICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "favicon.svg")

st.set_page_config(
    page_title="Induseuro-Labs Engineering Console",
    page_icon=FAVICON_PATH if os.path.exists(FAVICON_PATH) else None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Strip default Streamlit chrome completely
st.markdown("""
<style>
  /* Strip all default Streamlit chrome */
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

# Render the tactile Skeuomorphic / Claymorphic console
components.iframe(src="http://127.0.0.1:8000", height=960, scrolling=True)
