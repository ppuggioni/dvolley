from __future__ import annotations

import pandas as pd
import streamlit as st

from dvolley.domain.bayesian_stats import format_ci_range
from dvolley.domain.descriptive_touch_stats import (
    build_attack_quality_drilldown_table,
    build_descriptive_touch_stats,
    get_event_display_label,
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


def _format_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        low_suffix = " 95% CI low"
        high_suffix = " 95% CI high"
        drop_cols = []
        for col in list(out.columns):
            metric_name = str(col[1]) if isinstance(col, tuple) and len(col) > 1 else str(col)
            if not metric_name.endswith(low_suffix):
                continue
            prefix = metric_name[: -len(low_suffix)]
            high_col = (col[0], f"{prefix}{high_suffix}")
            if high_col not in out.columns:
                continue
            ci_col = (col[0], f"{prefix} CI")
            out[ci_col] = [format_ci_range(lo, hi) for lo, hi in zip(out[col], out[high_col])]
            drop_cols.extend([col, high_col])
        if drop_cols:
            out = out.drop(columns=drop_cols)

        preferred_metric_order = [
            "Actions",
            "% share",
            "% share CI",
            "Successful",
            "% successful",
            "% successful CI",
        ]
        ordered_cols = []
        segments = list(dict.fromkeys(out.columns.get_level_values(0).tolist()))
        for segment in segments:
            segment_metrics = [c[1] for c in out.columns if c[0] == segment]
            used_metrics = []
            for metric in preferred_metric_order:
                if metric in segment_metrics:
                    ordered_cols.append((segment, metric))
                    used_metrics.append(metric)
            for metric in segment_metrics:
                if metric not in used_metrics:
                    ordered_cols.append((segment, metric))
        out = out.reindex(columns=ordered_cols)

    for col in out.columns:
        metric_name = col[1] if isinstance(col, tuple) and len(col) > 1 else str(col)
        if str(metric_name).endswith(" CI"):
            continue
        if "%" in str(metric_name):
            out[col] = out[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
        else:
            out[col] = out[col].apply(lambda x: int(x) if pd.notna(x) else 0)
    return out


def _display_with_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = [get_event_display_label(str(idx)) for idx in out.index]
    out.index.name = "Event type"
    return out


def page_descriptive_stats_main(loader):
    st.title("Descriptive Statistics")
    st.info(
        "Descriptive touch-by-touch statistics for a selected team.\n"
        "- Sideout: selected team receives.\n"
        "- Breakpoint: selected team serves.\n"
        "- Shares in P1..P6 columns are within each rotation."
    )

    mode_label = st.radio(
        "Phase",
        options=["Sideout", "Breakpoint"],
        horizontal=True,
        key="descriptive_phase",
    )
    mode = mode_label.lower()

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
        key="descriptive_team_id",
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
        key="descriptive_match_selection_mode",
    )
    if selection_mode == "All":
        selected_match_ids = all_match_ids
    else:
        selected_match_ids = st.multiselect(
            "Select matches",
            options=all_match_ids,
            default=all_match_ids[: min(5, len(all_match_ids))],
            format_func=lambda mid: match_id_to_label.get(mid, mid),
            key="descriptive_match_ids",
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

    st.markdown("### Options")
    include_by_rotation = st.checkbox(
        "Show rotation breakdown (Total + P1..P6)",
        value=True,
        key="descriptive_include_by_rotation",
    )
    exclude_sideout_serve_errors = False
    if mode == "sideout":
        exclude_sideout_serve_errors = st.checkbox(
            "Exclude opponent serve errors from sideout stats",
            value=False,
            key="descriptive_exclude_sideout_serve_errors",
        )

    result = build_descriptive_touch_stats(
        touches_df=touches_df,
        team_id=selected_team_id,
        mode=mode,
        selected_match_ids=list(match_ids_cache_key),
        include_by_rotation=include_by_rotation,
        exclude_sideout_serve_errors=exclude_sideout_serve_errors,
    )

    if result.rallies_df.empty:
        st.warning("No valid rallies found for this selection.")
        return

    st.markdown("### Event Summary")
    st.caption(
        "Columns report Actions, share within segment, Successful points, success rate, and Bayesian 95% CI "
        "(Beta(1,1) prior)."
    )
    summary_display = _display_with_labels(_format_stats_table(result.summary_table))
    st.dataframe(summary_display, use_container_width=True)

    if not result.event_keys:
        st.info("No event rows available for attack-quality drilldown.")
        return

    selected_event = st.selectbox(
        "Event type for attack-quality breakdown",
        options=result.event_keys,
        index=0,
        format_func=get_event_display_label,
        key="descriptive_selected_event",
    )

    drilldown = build_attack_quality_drilldown_table(
        rallies_df=result.rallies_df,
        event_key=selected_event,
        include_by_rotation=include_by_rotation,
    )
    st.markdown(f"### Attack-Quality Breakdown for '{get_event_display_label(selected_event)}'")
    if drilldown.empty:
        st.info("No attack-quality rows available for this event.")
        return
    st.dataframe(_format_stats_table(drilldown), use_container_width=True)
