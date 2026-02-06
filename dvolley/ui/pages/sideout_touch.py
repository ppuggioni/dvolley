from __future__ import annotations

import pandas as pd
import streamlit as st

from dvolley.domain.breakpoint_touch_analysis import extract_team_catalog, extract_team_matches
from dvolley.domain.sideout_touch_analysis import SideoutTouchResult, build_sideout_touch_analysis
from dvolley.services.data_loader import load_full_data_from_db


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


def _format_percent_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if "%" in str(col):
            out[col] = out[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
    return out


def _validate_touch_schema(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_TOUCH_COLUMNS if c not in df.columns]


def _render_top_matrices(result: SideoutTouchResult):
    st.markdown("### Sideout Points/Actions Matrix by Evaluation")
    matrix_pa = result.matrix_points_and_actions
    if not matrix_pa.empty:
        st.dataframe(matrix_pa, use_container_width=True)

    st.markdown("### Action Counts by Evaluation")
    matrix_count = result.matrix_code_counts
    if not matrix_count.empty:
        st.dataframe(matrix_count, use_container_width=True)


def _render_summary_tables(result: SideoutTouchResult):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### Actions, Points, Opponent Serve Errors by Rotation")
        summary = _format_percent_cols(result.rotation_summary)
        st.dataframe(summary, use_container_width=True)

    with c2:
        st.markdown("### First Attack After Reception by Player")
        attackers = _format_percent_cols(result.attacker_summary.copy())
        st.dataframe(attackers, use_container_width=True)

    with c3:
        st.markdown("### Class Summary")
        class_df = _format_percent_cols(result.class_summary.copy())
        st.dataframe(class_df, use_container_width=True)


def _render_class_tables(result: SideoutTouchResult):
    st.markdown("### Class by Rotation Details")
    if not result.class_rotation_tables:
        st.info("No detail tables available.")
        return

    class_names = [c for c in result.class_order if c in result.class_rotation_tables]
    cols = st.columns(3)
    for i, class_name in enumerate(class_names):
        with cols[i % 3]:
            st.markdown(f"**{class_name}**")
            table = result.class_rotation_tables[class_name].copy()
            table["%"] = table["%"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
            st.dataframe(table, use_container_width=True)


def render_sideout_analysis_from_touches(
    touches_df: pd.DataFrame,
    selected_team_id: str,
    selected_match_ids: list[str] | None = None,
    show_title: bool = True,
):
    if show_title:
        st.title("Sideout Analysis (Touch-by-Touch)")
    else:
        st.markdown("### Sideout Analysis (Touch-by-Touch)")
    st.caption(
        "Sideout phase analysis: the selected team must be receiving. "
        "Opponent ACE is counted when the serving team scores with serve evaluation '#', "
        "or when there is no reception and the rally ends immediately."
    )

    if touches_df.empty:
        st.warning("No touch-by-touch data available in the database.")
        return

    missing_cols = _validate_touch_schema(touches_df)
    if missing_cols:
        st.error(f"Missing required touch columns: {missing_cols}")
        return

    result = build_sideout_touch_analysis(
        touches_df,
        selected_team_id,
        selected_match_ids=selected_match_ids,
    )

    if result.receive_rallies.empty:
        st.warning("No valid rallies found for the current selection.")
        return

    total_rallies = len(result.receive_rallies)
    action_mask = result.receive_rallies["class_label"] != "OPPONENT SERVE ERROR"
    total_actions = int(action_mask.sum())
    total_points = int(result.receive_rallies.loc[action_mask, "sideout_point"].sum())
    total_opp_errors = int((~action_mask).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rallies", total_rallies)
    c2.metric("Actions", total_actions)
    c3.metric("Sideout points", total_points)
    c4.metric("Opponent serve errors", total_opp_errors)

    with st.expander("Rally count diagnostics", expanded=False):
        st.write(result.diagnostics)

    unknown_rows = (result.receive_rallies["class_label"] == "OUR RECEPTION OTHER").sum()
    if unknown_rows:
        st.warning(
            f"{unknown_rows} rallies were not classified in standard classes "
            "(shown as 'OUR RECEPTION OTHER')."
        )

    _render_top_matrices(result)
    _render_summary_tables(result)
    _render_class_tables(result)


def page_sideout_touch_main(show_title: bool = True):
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
        key="so_touch_team_id",
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
        key="so_touch_selection_mode",
    )
    if selection_mode == "All":
        selected_match_ids = all_match_ids
    else:
        selected_match_ids = st.multiselect(
            "Select matches",
            options=all_match_ids,
            default=all_match_ids[: min(5, len(all_match_ids))],
            format_func=lambda mid: match_id_to_label.get(mid, mid),
            key="so_touch_match_ids",
        )

    if not selected_match_ids:
        st.info("Select at least one match.")
        return

    render_sideout_analysis_from_touches(
        touches_df,
        selected_team_id,
        selected_match_ids=selected_match_ids,
        show_title=show_title,
    )
