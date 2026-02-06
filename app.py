import logging
import threading
from typing import Optional
import os
import sys

import pandas as pd
import streamlit as st

# Ensure local package imports work in Streamlit Cloud and local runs
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dvolley.services.data_loader import load_data_from_db
from dvolley.ui.pages.setup import fit_selected_model
from dvolley.ui.pages.model_analysis import page_model_analysis
from dvolley.ui.pages.detailed_analysis import page_detailed_analysis_main
from dvolley.ui.pages.rotation import page_rotation_main
from dvolley.ui.pages.teams_summary import page_teams_summary
from dvolley.ui.pages.wip import wip_page_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

PAGE_SETUP = "Setup & Status"
PAGE_ROTATION = "Rotation Simulator"
PAGE_TEAMS_SUMMARY = "Teams Summary"
PAGE_MODEL_ANALYSIS = "Model Analysis"
PAGE_DETAILED_ANALYSIS = "Detailed Analysis"
PAGE_WIP = "Work in Progress"
DEFAULT_AUTO_MODEL = "logistic_rotation_alpha_0.005"


class BackgroundLoader:
    def __init__(self):
        self.data = None
        self.is_loading = False
        self.thread = None
        self.error = None
        self.progress_text = ""
        self.load_id = 0

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
            if df is not None:
                self.load_id += 1
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
    st.session_state.setdefault("fit_model_option", DEFAULT_AUTO_MODEL)

    perform_load_async()

    loader = get_loader()
    if loader.is_loading:
        st.sidebar.info(f"⏳ Loading data... {loader.progress_text}")
    elif loader.data is not None:
        if "loaded_matches_df" not in st.session_state:
            st.session_state["loaded_matches_df"] = loader.data
        st.sidebar.success(f"✅ Data loaded ({len(loader.data)} rallies)")

        last_fit_load_id = st.session_state.get("last_auto_fit_load_id")
        if last_fit_load_id != loader.load_id:
            selected_model = st.session_state.get("fit_model_option", DEFAULT_AUTO_MODEL)
            try:
                success = fit_selected_model(loader.data, selected_model, show_messages=False)
                st.session_state["last_auto_fit_load_id"] = loader.load_id
                st.session_state["auto_fit_error"] = None if success else "Auto-fit failed."
            except Exception as e:
                st.session_state["last_auto_fit_load_id"] = loader.load_id
                st.session_state["auto_fit_error"] = str(e)

    st.sidebar.divider()
    if "fitted_params_df" in st.session_state:
        st.sidebar.success("🔧 Using refitted parameters")
    else:
        st.sidebar.info("📁 No parameters fitted yet")

    st.sidebar.title("Menu")
    page = st.sidebar.selectbox(
        "Select page",
        options=[
            PAGE_SETUP,
            PAGE_DETAILED_ANALYSIS,
            PAGE_ROTATION,
            PAGE_TEAMS_SUMMARY,
            PAGE_MODEL_ANALYSIS,
            PAGE_WIP,
        ],
        index=0,
    )

    last_match_date = get_last_match_date(loader.data)

    st.session_state["perform_load_async"] = perform_load_async

    if page == PAGE_SETUP:
        from dvolley.ui.pages.setup import page_setup_status
        page_setup_status(loader, last_match_date=last_match_date)
    elif page == PAGE_DETAILED_ANALYSIS:
        page_detailed_analysis_main(loader)
    elif page == PAGE_ROTATION:
        page_rotation_main(loader, last_match_date=last_match_date)
    elif page == PAGE_TEAMS_SUMMARY:
        page_teams_summary(loader, last_match_date=last_match_date)
    elif page == PAGE_MODEL_ANALYSIS:
        page_model_analysis()
    else:
        wip_page_main()


if __name__ == "__main__":
    main()
