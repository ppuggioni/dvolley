from __future__ import annotations

import pandas as pd
import streamlit as st

from dvolley.domain.bayesian_stats import format_ci_range
from dvolley.domain.player_analysis import (
    PASS_QUALITY_ORDER,
    PlayerAnalysisTables,
    build_player_analysis_tables,
    build_player_sideout_dataset,
)
from dvolley.services.data_loader import load_matches_data_from_db
from dvolley.ui.coloring import build_style_matrix


QUALITY_LABELS = {"OTHER": "Other"}


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


def _prepare_stats_table(df: pd.DataFrame) -> pd.DataFrame:
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
            "Rallies won",
            "% rally won",
            "% rally won CI",
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
    return out


def _build_stats_table_styler(
    display_df: pd.DataFrame,
    raw_df: pd.DataFrame,
):
    if not isinstance(display_df.columns, pd.MultiIndex):
        return display_df

    def baseline_fn(raw: pd.DataFrame, idx: object, col: object) -> object:
        if not isinstance(col, tuple) or len(col) != 2:
            return None
        segment, metric = col
        if metric in {"% successful", "% rally won"}:
            total_col = ("Total", metric)
            if segment != "Total" and total_col in raw.columns:
                return raw.at[idx, total_col]
            if "Grand total" in raw.index and (segment, metric) in raw.columns:
                return raw.at["Grand total", (segment, metric)]
            return None
        if metric == "% share":
            if segment == "Total" or ("Total", "% share") not in raw.columns:
                return None
            return raw.at[idx, ("Total", "% share")]
        return None

    def ci_column_fn(col: object) -> tuple[object | None, object | None]:
        if not isinstance(col, tuple) or len(col) != 2:
            return (None, None)
        segment, metric = col
        return ((segment, f"{metric} 95% CI low"), (segment, f"{metric} 95% CI high"))

    target_cols = [
        col
        for col in display_df.columns
        if isinstance(col, tuple) and len(col) == 2 and col[1] in {"% share", "% successful", "% rally won"}
    ]
    style_matrix = build_style_matrix(
        display_df,
        raw_df,
        columns=target_cols,
        baseline_fn=baseline_fn,
        ci_column_fn=ci_column_fn,
        skip_rows={"Grand total"},
    )
    formatters = {}
    for col in display_df.columns:
        metric_name = col[1] if isinstance(col, tuple) and len(col) > 1 else str(col)
        if str(metric_name).endswith("CI"):
            continue
        if "%" in str(metric_name):
            formatters[col] = lambda x: f"{x:.2%}" if pd.notna(x) else "-"
        else:
            formatters[col] = lambda x: int(x) if pd.notna(x) else 0
    return display_df.style.apply(lambda _: style_matrix, axis=None).format(formatters)


def _label_index(df: pd.DataFrame, *, index_name: str) -> pd.DataFrame:
    out = df.copy()
    out.index = [QUALITY_LABELS.get(str(idx), str(idx)) for idx in out.index]
    out.index.name = index_name
    return out


def _render_stats_table(df: pd.DataFrame, *, index_name: str):
    if df.empty:
        st.info("No rows available for this selection.")
        return
    labeled_raw = _label_index(df, index_name=index_name)
    display = _prepare_stats_table(labeled_raw)
    styler = _build_stats_table_styler(display, labeled_raw)
    st.dataframe(styler, use_container_width=True)


def _render_first_attack_by_pass_tables(result: PlayerAnalysisTables):
    if not result.first_attack_by_pass_quality:
        st.info("No first-attacks found for this player.")
        return

    ordered_passes = [code for code in PASS_QUALITY_ORDER if code in result.first_attack_by_pass_quality]
    for pass_quality in result.first_attack_by_pass_quality:
        if pass_quality not in ordered_passes:
            ordered_passes.append(pass_quality)

    tabs = st.tabs([f"Pass {QUALITY_LABELS.get(code, code)}" for code in ordered_passes])
    for tab, pass_quality in zip(tabs, ordered_passes):
        with tab:
            _render_stats_table(
                result.first_attack_by_pass_quality[pass_quality],
                index_name="Attack quality",
            )


def page_player_analysis_main(loader):
    st.title("Player Analysis")
    st.info(
        "Sideout-only player analysis.\n"
        "- First sideout attack quality by pass quality and rotation (+ aggregates).\n"
        "- Non-first attack quality by rotation (+ aggregates), using attacks from both sideout and breakpoint rallies.\n"
        "- First-pass quality by rotation (+ aggregates).\n"
        "- Team and match lists come from rally data; touch data is loaded only for selected matches."
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
        key="player_analysis_team_id",
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
        key="player_analysis_match_selection_mode",
    )
    if selection_mode == "All":
        selected_match_ids = all_match_ids
    else:
        selected_match_ids = st.multiselect(
            "Select matches",
            options=all_match_ids,
            default=all_match_ids[: min(5, len(all_match_ids))],
            format_func=lambda mid: match_id_to_label.get(mid, mid),
            key="player_analysis_match_ids",
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
    with st.spinner("Building sideout player dataset..."):
        dataset = build_player_sideout_dataset(
            touches_df=touches_df,
            team_id=selected_team_id,
            selected_match_ids=list(match_ids_cache_key),
        )

    if dataset.sideout_rallies.empty:
        st.warning("No sideout rallies found for this selection.")
        return
    if not dataset.players:
        st.warning("No players found for this selection.")
        return

    selected_player = st.selectbox(
        "Select player",
        options=dataset.players,
        index=None,
        placeholder="Choose a player...",
        key="player_analysis_player_name",
    )
    if not selected_player:
        st.info("Select a player to generate analysis.")
        return

    include_by_rotation = st.checkbox(
        "Show rotation breakdown (Total + P1..P6)",
        value=True,
        key="player_analysis_include_by_rotation",
    )

    result = build_player_analysis_tables(
        dataset,
        selected_player,
        include_by_rotation=include_by_rotation,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sideout rallies", int(len(dataset.sideout_rallies)))
    c2.metric("Player first attacks", int(result.first_attack_attempts))
    c3.metric("Player non-first attacks", int(result.non_first_attack_attempts))
    c4.metric("Player first passes", int(result.pass_attempts))

    st.markdown("### 1) First-Attack Quality in Sideout")
    st.caption(
        "Rows = first-attack quality for the selected player. "
        "Outcome columns report rallies won by the player's team."
    )
    _render_stats_table(result.first_attack_overall, index_name="Attack quality")

    st.markdown("#### First Attack by Pass Quality")
    st.caption("Each tab keeps the same metrics split by pass quality, with aggregate and rotation columns.")
    _render_first_attack_by_pass_tables(result)

    st.markdown("### 2) Non-First Attack Quality")
    st.caption(
        "Rows = attack quality for attacks that are not the first attack after reception, from both sideout and breakpoint rallies. "
        "Outcome columns report rallies won by the player's team."
    )
    _render_stats_table(result.non_first_attack_table, index_name="Attack quality")

    st.markdown("### 3) Pass Quality in Sideout")
    st.caption(
        "Rows = first-pass quality for rallies where this player receives serve first. "
        "Outcome columns report rallies won by the player's team."
    )
    _render_stats_table(result.pass_quality_table, index_name="Pass quality")

    with st.expander("Diagnostics", expanded=False):
        st.write(dataset.diagnostics)
