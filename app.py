import logging
import threading
from typing import Optional

import pandas as pd
import streamlit as st

from dvolley.services.data_loader import load_data_from_db
from dvolley.ui.pages.load_data import page_load_data
from dvolley.ui.pages.model_analysis import page_model_analysis
from dvolley.ui.pages.rotation import page_rotation_main
from dvolley.ui.pages.teams_summary import page_teams_summary
from dvolley.ui.pages.wip import wip_page_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

PAGE_ROTATION = "rotation_simulator"
PAGE_TEAMS_SUMMARY = "teams_summary"
PAGE_LOAD_DATA = "load_data"
PAGE_MODEL_ANALYSIS = "model_analysis"
PAGE_WIP = "work in progress"


class BackgroundLoader:
    def __init__(self):
        self.data = None
        self.is_loading = False
        self.thread = None
        self.error = None
        self.progress_text = ""

    def start_loading(self):
        if not self.is_loading and self.data is None:
            self.is_loading = True
            self.progress_text = "Starting load from Database..."
            self.thread = threading.Thread(target=self._load)
            self.thread.start()

    def update_progress(self, current, total, message):
        self.progress_text = f"{message}"

    def _load(self):
        try:
            df = load_data_from_db()
            self.data = df
        except Exception as e:
            self.error = str(e)
        finally:
            self.is_loading = False


@st.cache_resource
def get_loader():
    return BackgroundLoader()


def perform_load_async():
    loader = get_loader()
    if loader.data is not None or loader.is_loading:
        return
    loader.start_loading()


def get_last_match_date(df: Optional[pd.DataFrame]) -> Optional[str]:
    if df is None or df.empty or "match_date" not in df.columns:
        return None
    dates = pd.to_datetime(df["match_date"], errors="coerce")
    if dates.isna().all():
        return None
    last_date = dates.max()
    return last_date.date().isoformat()


def main():
    st.set_page_config(page_title="Rotation App", layout="wide")

    perform_load_async()

    loader = get_loader()
    if loader.is_loading:
        st.sidebar.info(f"⏳ Loading data... {loader.progress_text}")
    elif loader.data is not None:
        if "loaded_matches_df" not in st.session_state:
            st.session_state["loaded_matches_df"] = loader.data
        st.sidebar.success(f"✅ Data loaded ({len(loader.data)} rallies)")

    st.sidebar.divider()
    if "fitted_params_df" in st.session_state:
        st.sidebar.success("🔧 Using refitted parameters")
    else:
        st.sidebar.info("📁 No parameters fitted yet")

    st.sidebar.title("Menu")
    page = st.sidebar.selectbox(
        "Select page",
        options=[PAGE_ROTATION, PAGE_TEAMS_SUMMARY, PAGE_MODEL_ANALYSIS, PAGE_LOAD_DATA, PAGE_WIP],
        index=0,
    )

    last_match_date = get_last_match_date(loader.data)

    st.session_state["perform_load_async"] = perform_load_async

    if page == PAGE_ROTATION:
        page_rotation_main(loader, last_match_date=last_match_date)
    elif page == PAGE_TEAMS_SUMMARY:
        page_teams_summary(loader, last_match_date=last_match_date)
    elif page == PAGE_MODEL_ANALYSIS:
        page_model_analysis()
    elif page == PAGE_LOAD_DATA:
        page_load_data(loader, last_match_date=last_match_date)
    else:
        wip_page_main()


if __name__ == "__main__":
    main()
