from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dvolley.domain.bayesian_stats import format_ci_range
from dvolley.domain.conditional_breakpoint_analysis import (
    ConditionalBreakpointResult,
    build_conditional_breakpoint_analysis,
)
from dvolley.services.data_loader import load_matches_data_from_db
from dvolley.ui.coloring import build_style_matrix


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


def _prepare_probability_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    low_suffix = " 95% CI low"
    high_suffix = " 95% CI high"
    for col in list(out.columns):
        col_name = str(col)
        if not col_name.endswith(low_suffix):
            continue
        prefix = col_name[: -len(low_suffix)]
        high_col = f"{prefix}{high_suffix}"
        if high_col not in out.columns:
            continue
        ci_col = f"{prefix} CI"
        out[ci_col] = [format_ci_range(lo, hi) for lo, hi in zip(out[col], out[high_col])]
        out = out.drop(columns=[col, high_col])
    return out


def _build_probability_styler(
    display_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    colored_cols: list[str],
    baseline_by_col: dict[str, object],
    skip_rows: set[str] | None = None,
):
    def baseline_fn(raw: pd.DataFrame, idx: object, col: object) -> object:
        if col not in baseline_by_col:
            return None
        spec = baseline_by_col[col]
        if callable(spec):
            return spec(raw, idx)
        return spec

    style_matrix = build_style_matrix(
        display_df,
        raw_df,
        columns=colored_cols,
        baseline_fn=baseline_fn,
        ci_column_fn=lambda col: (f"{col} 95% CI low", f"{col} 95% CI high"),
        skip_rows=(skip_rows or set()),
    )
    formatters = {}
    for col in display_df.columns:
        c = str(col).lower()
        if str(col).endswith(" CI"):
            continue
        if "probability" in c or "share" in c:
            formatters[col] = lambda x: f"{x:.2%}" if pd.notna(x) else "-"
        elif "attempts" in c or "count" in c:
            formatters[col] = lambda x: int(x) if pd.notna(x) else 0
    return display_df.style.apply(lambda _: style_matrix, axis=None).format(formatters)


def _format_probability_with_ci(prob: object, low: object, high: object) -> str:
    if pd.isna(prob):
        return "-"
    return f"{prob:.2%} {format_ci_range(low, high)}"


def _build_rotation_probability_with_ci_table(result: ConditionalBreakpointResult) -> pd.DataFrame:
    source = result.rotation_quality_summary
    if source is None or source.empty:
        return pd.DataFrame()

    prob_pivot = source.pivot(index="Rotation", columns="Attack_quality", values="Point_won_probability")
    low_pivot = source.pivot(index="Rotation", columns="Attack_quality", values="Point_won_probability 95% CI low")
    high_pivot = source.pivot(index="Rotation", columns="Attack_quality", values="Point_won_probability 95% CI high")
    if result.rotation_probability_pivot is not None and not result.rotation_probability_pivot.empty:
        prob_pivot = prob_pivot.reindex(
            index=result.rotation_probability_pivot.index,
            columns=result.rotation_probability_pivot.columns,
        )
        low_pivot = low_pivot.reindex(
            index=result.rotation_probability_pivot.index,
            columns=result.rotation_probability_pivot.columns,
        )
        high_pivot = high_pivot.reindex(
            index=result.rotation_probability_pivot.index,
            columns=result.rotation_probability_pivot.columns,
        )

    baseline_by_quality = {
        str(row["Attack_quality"]): row["Point_won_probability"]
        for _, row in result.quality_summary.iterrows()
        if row["Attack_quality"] != "Total"
    }
    display = prob_pivot.copy().astype(object)
    for col in display.columns:
        for idx in display.index:
            display.at[idx, col] = _format_probability_with_ci(
                prob_pivot.at[idx, col],
                low_pivot.at[idx, col],
                high_pivot.at[idx, col],
            )
    raw = prob_pivot.copy()
    for col in prob_pivot.columns:
        raw[f"{col} 95% CI low"] = low_pivot[col]
        raw[f"{col} 95% CI high"] = high_pivot[col]

    style_matrix = build_style_matrix(
        display,
        raw,
        columns=list(prob_pivot.columns),
        baseline_fn=lambda _raw_df, _idx, col: baseline_by_quality.get(str(col)),
        ci_column_fn=lambda col: (f"{col} 95% CI low", f"{col} 95% CI high"),
        skip_rows=set(),
    )
    return display.style.apply(lambda _: style_matrix, axis=None)


def _render_main_tables(result: ConditionalBreakpointResult, mode: str):
    quality_prob_baseline = (
        result.quality_summary.loc[
            result.quality_summary["Attack_quality"] == "Total",
            "Point_won_probability",
        ].iloc[0]
        if "Total" in set(result.quality_summary["Attack_quality"].tolist())
        else None
    )
    quality_share_baseline = {
        str(row["Attack_quality"]): row["Condition_share_of_first_attacks"]
        for _, row in result.quality_summary.iterrows()
        if row["Attack_quality"] != "Total"
    }

    st.markdown("### Conditional Point-Won Probability by First-Attack Quality")
    quality_raw = result.quality_summary.copy()
    quality_display = _prepare_probability_table(quality_raw)
    non_total_quality = max(len(quality_display[quality_display["Attack_quality"] != "Total"]), 1)
    quality_styler = _build_probability_styler(
        quality_display,
        quality_raw,
        colored_cols=["Point_won_probability", "Condition_share_of_first_attacks"],
        baseline_by_col={
            "Point_won_probability": quality_prob_baseline,
            "Condition_share_of_first_attacks": 1.0 / non_total_quality,
        },
        skip_rows={"Total"},
    )
    st.dataframe(quality_styler, use_container_width=True)

    st.markdown(f"### Breakdown by Rotation ({result.rotation_axis_label})")
    rotation_raw = result.rotation_quality_summary.copy()
    rotation_display = _prepare_probability_table(rotation_raw)
    quality_prob_by_quality = {
        str(row["Attack_quality"]): row["Point_won_probability"]
        for _, row in result.quality_summary.iterrows()
        if row["Attack_quality"] != "Total"
    }
    rotation_styler = _build_probability_styler(
        rotation_display,
        rotation_raw,
        colored_cols=[
            "Point_won_probability",
            "Condition_share_of_first_attacks",
            "Condition_share_within_rotation",
        ],
        baseline_by_col={
            "Point_won_probability": (
                lambda raw_df, idx: quality_prob_by_quality.get(str(raw_df.at[idx, "Attack_quality"]))
            ),
            "Condition_share_of_first_attacks": (
                lambda raw_df, idx: quality_share_baseline.get(str(raw_df.at[idx, "Attack_quality"]))
            ),
            "Condition_share_within_rotation": (
                lambda raw_df, idx: quality_share_baseline.get(str(raw_df.at[idx, "Attack_quality"]))
            ),
        },
        skip_rows=set(),
    )
    st.dataframe(rotation_styler, use_container_width=True)

    st.markdown("### Rotation x Attack Quality (Point-Won Probability + 95% CI)")
    st.dataframe(_build_rotation_probability_with_ci_table(result), use_container_width=True)

    if mode == "sideout":
        st.markdown("### Sideout only: Player Breakdown (First Attack)")
        if result.player_summary.empty:
            st.info("No player-level rows available for this selection.")
        else:
            player_raw = result.player_summary.copy()
            player_display = _prepare_probability_table(player_raw)
            player_count = max(len(player_display), 1)
            player_styler = _build_probability_styler(
                player_display,
                player_raw,
                colored_cols=["Point_won_probability", "Condition_share_of_first_attacks"],
                baseline_by_col={
                    "Point_won_probability": quality_prob_baseline,
                    "Condition_share_of_first_attacks": 1.0 / player_count,
                },
                skip_rows=set(),
            )
            st.dataframe(player_styler, use_container_width=True)
            st.markdown("#### Player x Attack Quality")
            player_quality_raw = result.player_quality_summary.copy()
            player_quality_display = _prepare_probability_table(player_quality_raw)
            quality_prob_by_quality = {
                str(row["Attack_quality"]): row["Point_won_probability"]
                for _, row in result.quality_summary.iterrows()
                if row["Attack_quality"] != "Total"
            }
            player_quality_styler = _build_probability_styler(
                player_quality_display,
                player_quality_raw,
                colored_cols=[
                    "Point_won_probability",
                    "Condition_share_of_first_attacks",
                    "Condition_share_within_player",
                ],
                baseline_by_col={
                    "Point_won_probability": (
                        lambda raw_df, idx: quality_prob_by_quality.get(str(raw_df.at[idx, "Attack_quality"]))
                    ),
                    "Condition_share_of_first_attacks": (
                        lambda raw_df, idx: quality_share_baseline.get(str(raw_df.at[idx, "Attack_quality"]))
                    ),
                    "Condition_share_within_player": (
                        lambda raw_df, idx: quality_share_baseline.get(str(raw_df.at[idx, "Attack_quality"]))
                    ),
                },
                skip_rows=set(),
            )
            st.dataframe(player_quality_styler, use_container_width=True)


def page_conditional_breakpoint_main(loader):
    st.title("Conditional Breakpoint Probability")
    with st.expander("How to read this page", expanded=True):
        st.markdown(
            "- Goal: estimate **P(point won by selected team | first receiving attack quality)**.\n"
            "- `Team sideout`: selected team is receiving.\n"
            "- `Team breakpoint`: selected team is serving.\n"
            "- Rallies without a first receiving attack are excluded.\n"
            "- `Condition_share_of_first_attacks` shows how frequent each quality is in the analyzed sample "
            "(for example, 400/887 = 45.1%).\n"
            "- `Condition_share_within_rotation` and `Condition_share_within_player` are local composition shares.\n"
            "- Point-won probabilities include Bayesian 95% credible intervals (Beta(1,1) prior)."
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
            "P(point won by selected team | first attack quality), with selected team receiving."
        )
    else:
        st.caption(
            "P(point won by selected team | first attack quality), with selected team serving."
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
    total_points = int(result.rally_df["selected_team_point_won"].sum())
    p_points = (total_points / total) if total > 0 else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Rallies analyzed", total)
    c2.metric("Points won by selected team", total_points)
    c3.metric("Overall point-won probability", f"{p_points:.2%}" if pd.notna(p_points) else "-")

    _render_main_tables(result, mode=mode)

    with st.expander("Diagnostics", expanded=False):
        st.write(result.diagnostics)
