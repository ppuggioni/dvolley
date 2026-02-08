from __future__ import annotations

import pandas as pd
import streamlit as st

from dvolley.domain.conditional_breakpoint_analysis import (
    ConditionalBreakpointResult,
    build_conditional_breakpoint_analysis,
)
from dvolley.services.data_loader import load_matches_data_from_db


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


def _format_probability_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        c = str(col)
        if "probability" in c.lower() or "share" in c.lower():
            out[col] = out[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
    return out


def _render_main_tables(result: ConditionalBreakpointResult, mode: str):
    st.markdown("### Conditional Breakpoint Probability by First-Attack Quality")
    st.dataframe(_format_probability_columns(result.quality_summary), use_container_width=True)

    st.markdown(f"### Breakdown by Rotation ({result.rotation_axis_label})")
    st.dataframe(_format_probability_columns(result.rotation_quality_summary), use_container_width=True)

    st.markdown("### Rotation x Attack Quality (Breakpoint Probability)")
    pivot_fmt = result.rotation_probability_pivot.copy()
    for col in pivot_fmt.columns:
        pivot_fmt[col] = pivot_fmt[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
    st.dataframe(pivot_fmt, use_container_width=True)

    if mode == "sideout":
        st.markdown("### Sideout only: Player Breakdown (First Attack)")
        if result.player_summary.empty:
            st.info("No player-level rows available for this selection.")
        else:
            st.dataframe(_format_probability_columns(result.player_summary), use_container_width=True)
            st.markdown("#### Player x Attack Quality")
            st.dataframe(_format_probability_columns(result.player_quality_summary), use_container_width=True)


def page_conditional_breakpoint_main(loader):
    st.title("Conditional Breakpoint Probability")
    with st.expander("How to read this page", expanded=True):
        st.markdown(
            "- Goal: estimate **P(Breakpoint | first receiving attack quality)**.\n"
            "- `Team sideout`: selected team is receiving; breakpoint means the opponent wins the rally.\n"
            "- `Team breakpoint`: selected team is serving; breakpoint means the selected team wins the rally.\n"
            "- Rallies without a first receiving attack are excluded.\n"
            "- `Condition_share_of_first_attacks` shows how frequent each quality is in the analyzed sample "
            "(for example, 400/887 = 45.1%).\n"
            "- `Condition_share_within_rotation` and `Condition_share_within_player` are local composition shares."
        )

    mode_label = st.radio(
        "Team phase",
        options=["Team sideout", "Team breakpoint"],
        horizontal=True,
        key="conditional_bp_mode",
    )
    mode = "sideout" if mode_label == "Team sideout" else "breakpoint"

    if mode == "sideout":
        st.caption(
            "P(Breakpoint | first attack quality) where selected team is receiving. "
            "Breakpoint means opponent wins the rally."
        )
    else:
        st.caption(
            "P(Breakpoint | first attack quality) where selected team is serving. "
            "Breakpoint means selected team wins the rally."
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
        "Select team",
        options=team_options,
        index=None,
        placeholder="Choose a team...",
        format_func=lambda tid: f"{team_names.get(tid, tid)} ({tid})",
        key="conditional_bp_team_id",
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
        key="conditional_bp_match_selection_mode",
    )
    if selection_mode == "All":
        selected_match_ids = all_match_ids
    else:
        selected_match_ids = st.multiselect(
            "Select matches",
            options=all_match_ids,
            default=all_match_ids[: min(5, len(all_match_ids))],
            format_func=lambda mid: match_id_to_label.get(mid, mid),
            key="conditional_bp_match_ids",
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

    result = build_conditional_breakpoint_analysis(
        touches_df=touches_df,
        team_id=selected_team_id,
        mode=mode,
        selected_match_ids=list(match_ids_cache_key),
    )

    if result.rally_df.empty:
        st.warning("No rallies with a first receiving attack found for this selection.")
        with st.expander("Diagnostics", expanded=False):
            st.write(result.diagnostics)
        return

    total = int(len(result.rally_df))
    total_bp = int(result.rally_df["bp_outcome"].sum())
    p_bp = (total_bp / total) if total > 0 else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Rallies analyzed", total)
    c2.metric("Breakpoint events", total_bp)
    c3.metric("Overall breakpoint probability", f"{p_bp:.2%}" if pd.notna(p_bp) else "-")

    _render_main_tables(result, mode=mode)

    with st.expander("Diagnostics", expanded=False):
        st.write(result.diagnostics)
