from __future__ import annotations

import pandas as pd
import streamlit as st

from dvolley.domain.bayesian_stats import format_ci_range
from dvolley.domain.breakpoint_touch_analysis import (
    BreakpointTouchResult,
    build_breakpoint_touch_analysis,
    extract_team_catalog,
    extract_team_matches,
)
from dvolley.services.data_loader import load_full_data_from_db
from dvolley.ui.coloring import build_style_matrix


REQUIRED_TOUCH_COLUMNS = [
    "match_alternative_id",
    "match_date",
    "home_team_id",
    "home_team",
    "visiting_team_id",
    "visiting_team",
    "set_number",
    "rally_number",
    "serving_team",
    "point_won_by",
    "skill",
    "team",
    "evaluation_code",
    "home_setter_position",
    "visiting_setter_position",
]


@st.cache_data(show_spinner=False)
def _load_touch_data_cached() -> pd.DataFrame:
    return load_full_data_from_db()


def _prepare_table(df: pd.DataFrame) -> pd.DataFrame:
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


def _build_table_styler(
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
        col_name = str(col)
        if col_name.endswith(" CI"):
            continue
        if "%" in col_name:
            formatters[col] = lambda x: f"{x:.2%}" if pd.notna(x) else "-"
        elif pd.api.types.is_numeric_dtype(display_df[col]):
            formatters[col] = lambda x: int(x) if pd.notna(x) else 0
    return display_df.style.apply(lambda _: style_matrix, axis=None).format(formatters)


def _validate_touch_schema(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_TOUCH_COLUMNS if c not in df.columns]


def _render_top_matrices(result: BreakpointTouchResult):
    st.markdown("### Points/Receptions Matrix by Evaluation")
    matrix_pr = result.matrix_points_and_receptions
    if not matrix_pr.empty:
        st.dataframe(matrix_pr, use_container_width=True)

    st.markdown("### Serve Counts by Evaluation")
    matrix_count = result.matrix_code_counts
    if not matrix_count.empty:
        st.dataframe(matrix_count, use_container_width=True)


def _render_summary_tables(result: BreakpointTouchResult):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### Serves, Break Points, Errors by Rotation")
        raw = result.rotation_summary.copy()
        display = _prepare_table(raw)
        styler = _build_table_styler(
            display,
            raw,
            colored_cols=["% break on serves", "% break excl errors"],
            baseline_by_col={
                "% break on serves": (
                    lambda raw_df, _: raw_df.at["Total", "% break on serves"]
                    if "Total" in raw_df.index
                    else None
                ),
                "% break excl errors": (
                    lambda raw_df, _: raw_df.at["Total", "% break excl errors"]
                    if "Total" in raw_df.index
                    else None
                ),
            },
            skip_rows={"Total"},
        )
        st.dataframe(styler, use_container_width=True)

    with c2:
        st.markdown("### Break Points by Server")
        raw = result.player_summary.copy()
        display = _prepare_table(raw)
        baseline = (
            (raw["Break_points"].sum() / raw["Serves"].sum())
            if not raw.empty and raw["Serves"].sum() > 0
            else None
        )
        styler = _build_table_styler(
            display,
            raw,
            colored_cols=["% break on serves"],
            baseline_by_col={"% break on serves": baseline},
            skip_rows=set(),
        )
        st.dataframe(styler, use_container_width=True)

    with c3:
        st.markdown("### Class Summary")
        raw = result.class_summary.copy()
        display = _prepare_table(raw)
        class_count = max(len([idx for idx in raw.index if idx != "Total"]), 1)
        equal_share = 1.0 / class_count
        styler = _build_table_styler(
            display,
            raw,
            colored_cols=["% points on serves", "% of total serves", "% of total points"],
            baseline_by_col={
                "% points on serves": (
                    lambda raw_df, _: raw_df.at["Total", "% points on serves"]
                    if "Total" in raw_df.index
                    else None
                ),
                "% of total serves": equal_share,
                "% of total points": equal_share,
            },
            skip_rows={"Total"},
        )
        st.dataframe(styler, use_container_width=True)


def _render_class_tables(result: BreakpointTouchResult):
    st.markdown("### Class by Rotation Details")
    if not result.class_rotation_tables:
        st.info("No detail tables available.")
        return

    class_names = [c for c in result.class_order if c in result.class_rotation_tables]
    cols = st.columns(3)
    for i, class_name in enumerate(class_names):
        with cols[i % 3]:
            st.markdown(f"**{class_name}**")
            raw = result.class_rotation_tables[class_name].copy()
            display = _prepare_table(raw)
            styler = _build_table_styler(
                display,
                raw,
                colored_cols=["%"],
                baseline_by_col={"%": (lambda raw_df, _: raw_df.at["Total", "%"] if "Total" in raw_df.index else None)},
                skip_rows={"Total"},
            )
            st.dataframe(styler, use_container_width=True)


def render_breakpoint_analysis_from_touches(
    touches_df: pd.DataFrame,
    selected_team_id: str,
    selected_match_ids: list[str] | None = None,
    show_title: bool = True,
):
    if show_title:
        st.title("Breakpoint Analysis (Touch-by-Touch)")
    else:
        st.markdown("### Breakpoint Analysis (Touch-by-Touch)")
    st.caption(
        "Breakpoint phase analysis: the selected team must be serving. "
        "ACE: point won by the serving team with serve evaluation '#', or with no opponent touch "
        "and immediate rally end."
    )
    st.caption("Probability columns include Bayesian 95% CI (Beta(1,1) prior).")

    if touches_df.empty:
        st.warning("No touch-by-touch data available in the database.")
        return

    missing_cols = _validate_touch_schema(touches_df)
    if missing_cols:
        st.error(f"Missing required touch columns: {missing_cols}")
        return

    result = build_breakpoint_touch_analysis(
        touches_df,
        selected_team_id,
        selected_match_ids=selected_match_ids,
    )

    if result.serve_rallies.empty:
        st.warning("No valid rallies found for the current selection.")
        return

    total_serves = len(result.serve_rallies)
    total_break = int(result.serve_rallies["break_point"].sum())
    total_errors = int(result.serve_rallies["serve_error"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Serves", total_serves)
    c2.metric("Break points", total_break)
    c3.metric("Errors", total_errors)

    with st.expander("Rally count diagnostics", expanded=False):
        st.write(result.diagnostics)

    unknown_rows = (result.serve_rallies["class_label"] == "OPP RECEPTION OTHER").sum()
    if unknown_rows:
        st.warning(
            f"{unknown_rows} rallies were not classified in standard classes "
            "(shown as 'OPP RECEPTION OTHER')."
        )

    _render_top_matrices(result)
    _render_summary_tables(result)
    _render_class_tables(result)


def page_breakpoint_touch_main(show_title: bool = True):
    with st.spinner("Loading touch-by-touch data from database..."):
        touches_df = _load_touch_data_cached()

    if touches_df.empty:
        st.warning("No touch-by-touch data available in the database.")
        return

    teams = extract_team_catalog(touches_df)
    if teams.empty:
        st.warning("No teams found in touch data.")
        return

    team_options = teams["team_id"].astype(str).tolist()
    team_names = {
        str(row["team_id"]): str(row["team_name"])
        for _, row in teams.iterrows()
    }

    selected_team_id = st.selectbox(
        "Select team to analyze",
        options=team_options,
        index=None,
        placeholder="Choose a team...",
        format_func=lambda tid: f"{team_names.get(tid, tid)} ({tid})",
        key="bp_touch_team_id",
    )
    if not selected_team_id:
        st.info("Select a team to load matches.")
        return

    matches = extract_team_matches(touches_df, selected_team_id).copy()
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
        key="bp_touch_selection_mode",
    )
    if selection_mode == "All":
        selected_match_ids = all_match_ids
    else:
        selected_match_ids = st.multiselect(
            "Select matches",
            options=all_match_ids,
            default=all_match_ids[: min(5, len(all_match_ids))],
            format_func=lambda mid: match_id_to_label.get(mid, mid),
            key="bp_touch_match_ids",
        )

    if not selected_match_ids:
        st.info("Select at least one match.")
        return

    render_breakpoint_analysis_from_touches(
        touches_df,
        selected_team_id,
        selected_match_ids=selected_match_ids,
        show_title=show_title,
    )
