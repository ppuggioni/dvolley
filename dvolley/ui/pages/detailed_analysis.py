from __future__ import annotations

import pandas as pd
import streamlit as st

from dvolley.services.data_loader import load_matches_data_from_db
from dvolley.ui.pages.breakpoint_touch import render_breakpoint_analysis_from_touches
from dvolley.ui.pages.sideout_touch import render_sideout_analysis_from_touches


@st.cache_data(show_spinner=False)
def _load_touches_for_matches_cached(match_ids: tuple[str, ...]) -> pd.DataFrame:
    return load_matches_data_from_db(list(match_ids))


def _extract_team_catalog_from_rallies(rallies_df: pd.DataFrame) -> pd.DataFrame:
    if rallies_df is None or rallies_df.empty:
        return pd.DataFrame(columns=["team_id", "team_name"])

    home = (
        rallies_df[["team_id_h", "team_h"]]
        .dropna(subset=["team_id_h"])
        .rename(columns={"team_id_h": "team_id", "team_h": "team_name"})
    )
    away = (
        rallies_df[["team_id_a", "team_a"]]
        .dropna(subset=["team_id_a"])
        .rename(columns={"team_id_a": "team_id", "team_a": "team_name"})
    )
    teams = pd.concat([home, away], ignore_index=True)
    teams["team_id"] = teams["team_id"].astype(str)
    teams["team_name"] = teams["team_name"].fillna("Unknown").astype(str)
    teams = teams.drop_duplicates(["team_id", "team_name"])
    return teams.sort_values(["team_name", "team_id"]).reset_index(drop=True)


def _extract_team_matches_from_rallies(rallies_df: pd.DataFrame, team_id: str) -> pd.DataFrame:
    if rallies_df is None or rallies_df.empty:
        return pd.DataFrame(
            columns=["match_alternative_id", "match_date", "home_team", "visiting_team", "label"]
        )

    team_id = str(team_id)
    df = rallies_df.copy()
    df["team_id_h"] = df["team_id_h"].astype(str)
    df["team_id_a"] = df["team_id_a"].astype(str)

    match_mask = (df["team_id_h"] == team_id) | (df["team_id_a"] == team_id)
    subset = df.loc[match_mask].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=["match_alternative_id", "match_date", "home_team", "visiting_team", "label"]
        )

    base = (
        subset.groupby("match_alternative_id", dropna=False)
        .agg(
            match_date=("match_date", "first"),
            home_team=("team_h", "first"),
            visiting_team=("team_a", "first"),
        )
        .reset_index()
    )
    base["match_date"] = base["match_date"].fillna("Unknown").astype(str)
    base["label"] = (
        base["match_date"].astype(str)
        + " | "
        + base["home_team"].fillna("Unknown").astype(str)
        + " vs "
        + base["visiting_team"].fillna("Unknown").astype(str)
    )
    return base.sort_values("match_date").reset_index(drop=True)


def page_detailed_analysis_main(loader):
    st.title("Detailed Analysis")
    mode = st.radio(
        "Phase",
        options=["Breakpoint", "Sideout"],
        horizontal=True,
        key="detailed_analysis_phase",
    )

    rallies_df = getattr(loader, "data", None)
    if rallies_df is None:
        if getattr(loader, "is_loading", False):
            st.info("Rally dataset is still loading. Please wait.")
        else:
            st.warning("Rally dataset is not available yet.")
        return
    if rallies_df.empty:
        st.warning("No rally data available.")
        return

    teams = _extract_team_catalog_from_rallies(rallies_df)
    if teams.empty:
        st.warning("No teams found in rally data.")
        return

    team_options = teams["team_id"].astype(str).tolist()
    team_names = {str(row["team_id"]): str(row["team_name"]) for _, row in teams.iterrows()}

    selected_team_id = st.selectbox(
        "Select team to analyze",
        options=team_options,
        index=None,
        placeholder="Choose a team...",
        format_func=lambda tid: f"{team_names.get(tid, tid)} ({tid})",
        key="detailed_team_id",
    )
    if not selected_team_id:
        st.info("Select a team to load matches.")
        return

    matches = _extract_team_matches_from_rallies(rallies_df, selected_team_id)
    if matches.empty:
        st.warning("No matches found for the selected team.")
        return

    st.markdown("### Team Matches")
    st.dataframe(
        matches[["match_date", "home_team", "visiting_team", "match_alternative_id"]],
        use_container_width=True,
    )

    match_id_to_label = {
        str(row["match_alternative_id"]): str(row["label"])
        for _, row in matches.iterrows()
    }
    all_match_ids = list(match_id_to_label.keys())

    selection_mode = st.radio(
        "Match selection",
        options=["All", "Manual selection"],
        horizontal=True,
        key="detailed_selection_mode",
    )
    if selection_mode == "All":
        selected_match_ids = all_match_ids
    else:
        selected_match_ids = st.multiselect(
            "Select matches",
            options=all_match_ids,
            default=all_match_ids[: min(5, len(all_match_ids))],
            format_func=lambda mid: match_id_to_label.get(mid, mid),
            key="detailed_match_ids",
        )

    if not selected_match_ids:
        st.info("Select at least one match.")
        return

    match_ids_cache_key = tuple(sorted(str(m) for m in selected_match_ids))
    with st.spinner("Loading touch-by-touch data for selected matches..."):
        touches_df = _load_touches_for_matches_cached(match_ids_cache_key)

    if touches_df.empty:
        st.warning("No touch-by-touch data found for selected matches.")
        return

    touches_df = touches_df[touches_df["match_alternative_id"].astype(str).isin(match_ids_cache_key)].copy()

    if mode == "Breakpoint":
        render_breakpoint_analysis_from_touches(
            touches_df,
            selected_team_id=selected_team_id,
            selected_match_ids=list(match_ids_cache_key),
            show_title=False,
        )
    else:
        render_sideout_analysis_from_touches(
            touches_df,
            selected_team_id=selected_team_id,
            selected_match_ids=list(match_ids_cache_key),
            show_title=False,
        )
