"""House-price analytics app — deployment entry point.

Two pages in one Streamlit process (DECISIONS.md D2: the live demo never
depends on a second running service):

  - Valuation tool        the deployed quantile model behind a simple form
  - Monitoring dashboard  Module 7's 2010 replay, rendered from committed
                          monitoring artifacts (Evidently never runs on the
                          free tier)
"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit Cloud clones into /mount/src/<repo>. If /mount is on sys.path,
# `import src` resolves to the mount folder, not this project's package.
_ROOT = Path(__file__).resolve().parent.parent
_SHADOW = {"/mount", "/mount/src"}
sys.path[:] = [
    p for p in sys.path
    if str(Path(p).resolve()) not in _SHADOW
]
sys.path.insert(0, str(_ROOT))

# set_page_config must be the FIRST Streamlit command — even the spinner of a
# cache_resource call before it makes Streamlit raise on startup
st.set_page_config(page_title="House Price Analytics", page_icon="🏠",
                   layout="wide")

pg = st.navigation([
    st.Page("valuation_tool.py", title="Valuation tool", icon="🏠",
            default=True),
    st.Page("monitoring_dashboard.py", title="Monitoring dashboard",
            icon="📈"),
])
pg.run()
