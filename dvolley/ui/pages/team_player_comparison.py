from __future__ import annotations

import pandas as pd
import streamlit as st

from dvolley.domain.player_analysis import PASS_QUALITY_ORDER, build_player_sideout_dataset
from dvolley.domain.team_player_comparison import (
    TeamPlayerComparisonResult,
    build_team_player_comparison,
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


def _build_comparison_table_styler(display_df: pd.DataFrame, raw_df: pd.DataFrame):
    if not isinstance(display_df.columns, pd.MultiIndex):
        return display_df

    def baseline_fn(raw: pd.DataFrame, idx: object, col: object) -> object:
        if idx == "TOTAL":
            return None
        if "TOTAL" not in raw.index:
            return None
        if not isinstance(col, tuple) or len(col) != 2:
            return None
        if col[0] == "Efficiency" and col[1] == "Score":
            return raw.at["TOTAL", col]
        if col[1] in {"% share", "% rally won"}:
            return raw.at["TOTAL", col]
        return None

    target_cols = [
        col
        for col in display_df.columns
        if isinstance(col, tuple)
        and len(col) == 2
        and ((col[0] == "Efficiency" and col[1] == "Score") or col[1] in {"% share", "% rally won"})
    ]
    style_matrix = build_style_matrix(
        display_df,
        raw_df,
        columns=target_cols,
        baseline_fn=baseline_fn,
        ci_column_fn=None,
        skip_rows={"TOTAL"},
    )

    formatters = {}
    for col in display_df.columns:
        if not isinstance(col, tuple) or len(col) != 2:
            continue
        metric = col[1]
        if metric == "Count":
            formatters[col] = lambda x: int(x) if pd.notna(x) else 0
        elif metric in {"% share", "% rally won"}:
            formatters[col] = lambda x: f"{x:.2%}" if pd.notna(x) else "-"
        elif col[0] == "Efficiency" and metric == "Score":
            formatters[col] = lambda x: f"{x:+.3f}" if pd.notna(x) else "-"

    return display_df.style.apply(lambda _: style_matrix, axis=None).format(formatters)


def _display_comparison_table(table: pd.DataFrame):
    if table.empty:
        st.info("No rows available for this selection.")
        return
    display = table.copy()
    renamed_cols = []
    for col in display.columns:
        if not isinstance(col, tuple) or len(col) != 2:
            renamed_cols.append(col)
            continue
        top = str(col[0])
        metric = str(col[1])
        renamed_cols.append((QUALITY_LABELS.get(top, top), metric))
    display.columns = pd.MultiIndex.from_tuples(renamed_cols)
    styler = _build_comparison_table_styler(display, display)
    st.dataframe(styler, use_container_width=True)


def _render_first_attack_by_pass_quality_tabs(result: TeamPlayerComparisonResult):
    if not result.first_attack_by_pass_quality:
        st.info("No sideout first-attack rows available for pass-quality splits.")
        return

    ordered_passes = [code for code in PASS_QUALITY_ORDER if code in result.first_attack_by_pass_quality]
    for pass_quality in result.first_attack_by_pass_quality:
        if pass_quality not in ordered_passes:
            ordered_passes.append(pass_quality)

    tabs = st.tabs([f"Pass {QUALITY_LABELS.get(code, code)}" for code in ordered_passes])
    for tab, pass_quality in zip(tabs, ordered_passes):
        with tab:
            _display_comparison_table(result.first_attack_by_pass_quality[pass_quality])


def page_team_player_comparison_main(loader):
    st.title("Team - Player Comparison")
    st.info(
        "Compare players for a selected team.\n"
        "1) Total attacks: rows are players (+ TOTAL); columns are attack-quality Count, % share, % rally won, "
        "plus Efficiency.\n"
        "2) First attack by pass quality: one table per pass quality, same structure.\n"
        "3) Pass quality: rows are players (+ TOTAL); columns are pass-quality Count, % share, % rally won, "
        "plus Efficiency.\n"
        "4) Serve quality: rows are players (+ TOTAL); columns are serve-quality Count, % share, % rally won, "
        "plus Efficiency.\n"
        "Efficiency = sum(share * (2 * rally_win_probability - 1))."
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
        key="team_player_comp_team_id",
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
        key="team_player_comp_match_selection_mode",
    )
    if selection_mode == "All":
        selected_match_ids = all_match_ids
    else:
        selected_match_ids = st.multiselect(
            "Select matches",
            options=all_match_ids,
            default=all_match_ids[: min(5, len(all_match_ids))],
            format_func=lambda mid: match_id_to_label.get(mid, mid),
            key="team_player_comp_match_ids",
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
    with st.spinner("Building team comparison tables..."):
        dataset = build_player_sideout_dataset(
            touches_df=touches_df,
            team_id=selected_team_id,
            selected_match_ids=list(match_ids_cache_key),
        )
        result = build_team_player_comparison(dataset)

    if dataset.team_attacks.empty and dataset.sideout_rallies.empty and dataset.team_serves.empty:
        st.warning("No attack/pass/serve rows found for the selected team and matches.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Team attacks", int(len(dataset.team_attacks)))
    c2.metric("Team sideout rallies", int(len(dataset.sideout_rallies)))
    c3.metric("Team serves", int(len(dataset.team_serves)))
    player_names = set(dataset.team_attacks["player_name"].dropna().astype(str).tolist())
    player_names.update(dataset.team_serves["player_name"].dropna().astype(str).tolist())
    c4.metric("Players", int(len(player_names)))

    st.markdown("### 1) Total Attacks")
    _display_comparison_table(result.total_attack_table)

    st.markdown("### 2) First Attack by Pass Quality")
    _render_first_attack_by_pass_quality_tabs(result)

    st.markdown("### 3) Pass Quality")
    _display_comparison_table(result.pass_quality_table)

    st.markdown("### 4) Serve Quality")
    _display_comparison_table(result.serve_quality_table)

    with st.expander("Diagnostics", expanded=False):
        st.write(dataset.diagnostics)
