import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Optional
from load_data import load_matches_from_drive
from load_full_data import process_dv_file_content
from gdrive_utils import read_file_content, read_file_bytes

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ------------------------------------------------------------
# App configuration / constants
# ------------------------------------------------------------
PAGE_ROTATION = "rotation_simulator"
PAGE_TEAMS_SUMMARY = "teams_summary"
PAGE_LOAD_DATA = "load_data"
PAGE_WIP = "work in progress"
PARAMS_FILE = "./params/params_out_break_sideout.csv"

POSITIONS = range(1, 7)
SLIDER_MIN = -2.0
SLIDER_MAX = 2.0
SLIDER_DEFAULT = 0.0
SLIDER_STEP = 0.01


# ------------------------------------------------------------
# Load CSV once
# ------------------------------------------------------------
@st.cache_data
def load_params(path: str = PARAMS_FILE) -> pd.DataFrame:
    """Load the whole params file (global + team)."""
    return pd.read_csv(path, dtype={"team_id": str})


def get_team_params(df_all: pd.DataFrame) -> pd.DataFrame:
    return df_all[df_all["par_type"] == "team"].copy()


def get_global_breakpoint_default(df_all: pd.DataFrame) -> float:
    # in the file it's called global_breakpoint
    m = (df_all["par_type"] == "global") & (df_all["par_name"] == "global_breakpoint")
    subset = df_all[m]
    if len(subset):
        return float(subset.iloc[0]["par_value"])
    return 0.0


# ------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------
def slider_sidebar(label: str, key: str, component=None):
    target = component if component is not None else st.sidebar
    current_val = st.session_state.get(key, SLIDER_DEFAULT)
    return target.slider(
        label,
        min_value=SLIDER_MIN,
        max_value=SLIDER_MAX,
        value=float(current_val),
        step=SLIDER_STEP,
        key=key,
    )


def reset_team_sliders(prefix: str):
    """Set all that team's sliders to 0."""
    st.session_state[f"{prefix}_bp_adjustment"] = 0.0
    st.session_state[f"{prefix}_so_adjustment"] = 0.0
    for pos in POSITIONS:
        st.session_state[f"{prefix}_pos{pos}_bp_adjustment"] = 0.0
        st.session_state[f"{prefix}_pos{pos}_so_adjustment"] = 0.0


def apply_team_preset_if_changed(
    prefix: str,
    selected_team_id: str | None,
    team_params_df: pd.DataFrame,
):
    """
    When user selects a team from the dropdown, load all its parameters
    from the CSV into the sliders.
    """
    prev_key = f"{prefix}_selected_team_id_prev"
    prev_val = st.session_state.get(prev_key)

    if not selected_team_id or selected_team_id == "Reset":
        st.session_state[prev_key] = selected_team_id
        return

    if selected_team_id != prev_val:
        rows = team_params_df[team_params_df["team_id"] == selected_team_id]
        for _, row in rows.iterrows():
            par_name = row["par_name"]
            par_value = float(row["par_value"])

            # team level
            if par_name == "breakpoint_team_adjustment":
                st.session_state[f"{prefix}_bp_adjustment"] = par_value
            elif par_name == "sideout_team_adjustment":
                st.session_state[f"{prefix}_so_adjustment"] = par_value

            # rotation level
            elif par_name.startswith("breakpoint_pos_"):
                pos = par_name.split("_")[-1]
                st.session_state[f"{prefix}_pos{pos}_bp_adjustment"] = par_value
            elif par_name.startswith("sideout_pos_"):
                pos = par_name.split("_")[-1]
                st.session_state[f"{prefix}_pos{pos}_so_adjustment"] = par_value

        st.session_state[prev_key] = selected_team_id


def render_team_block_sidebar(
    team_prefix: str,
    team_name: str,
    team_params_df: pd.DataFrame,
):
    st.sidebar.markdown(f"### {team_name}")

    # dropdown
    unique_teams = (
        team_params_df[["team_id", "team_name"]]
        .drop_duplicates()
        .sort_values("team_name")
    )
    options = ["Reset"] + [
        f"{row.team_id} - {row.team_name}" for _, row in unique_teams.iterrows()
    ]

    selected_option = st.sidebar.selectbox(
        f"Preset for {team_name}",
        options=options,
        key=f"{team_prefix}_preset_select",
    )

    prev_sel = st.session_state.get(f"{team_prefix}_selected_team_id_prev")

    if selected_option == "Reset":
        if prev_sel != "Reset":
            reset_team_sliders(team_prefix)
        st.session_state[f"{team_prefix}_current_team_id"] = "Reset"
        st.session_state[f"{team_prefix}_current_team_name"] = "Reset"
        st.session_state[f"{team_prefix}_selected_team_id_prev"] = "Reset"
    else:
        selected_team_id, selected_team_name = selected_option.split(" - ", 1)
        st.session_state[f"{team_prefix}_current_team_id"] = selected_team_id
        st.session_state[f"{team_prefix}_current_team_name"] = selected_team_name
        apply_team_preset_if_changed(
            team_prefix, selected_team_id, team_params_df
        )

    sb_left, sb_right = st.sidebar.columns(2)

    # BP (breakpoint)
    sb_left.markdown("**BP adjustments**")
    slider_sidebar(
        f"{team_name} BP (team)",
        key=f"{team_prefix}_bp_adjustment",
        component=sb_left,
    )
    for pos in POSITIONS:
        slider_sidebar(
            f"{team_name} pos{pos} BP",
            key=f"{team_prefix}_pos{pos}_bp_adjustment",
            component=sb_left,
        )

    # SO (sideout)
    sb_right.markdown("**SO adjustments**")
    slider_sidebar(
        f"{team_name} SO (team)",
        key=f"{team_prefix}_so_adjustment",
        component=sb_right,
    )
    for pos in POSITIONS:
        slider_sidebar(
            f"{team_name} pos{pos} SO",
            key=f"{team_prefix}_pos{pos}_so_adjustment",
            component=sb_right,
        )


def rotation_simulator_controls_in_sidebar(
    team_params_df: pd.DataFrame, global_breakpoint_default: float
):
    """
    Sidebar controls with APPLY at the top.
    """
    st.sidebar.markdown("## Rotation simulator")

    # APPLY at the top
    if st.sidebar.button("APPLY", type="primary", width='stretch'):
        run_simulation_and_store()

    # seed global_breakpoint once
    if "global_breakpoint" not in st.session_state:
        st.session_state["global_breakpoint"] = global_breakpoint_default

    # tiebreak toggle
    st.sidebar.checkbox("Tiebreak", key="tiebreak")

    # scores
    c1, c2 = st.sidebar.columns(2)
    with c1:
        st.number_input(
            "score_team_a",
            min_value=0,
            step=1,
            value=int(st.session_state.get("score_team_a", 0)),
            key="score_team_a",
        )
    with c2:
        st.number_input(
            "score_team_b",
            min_value=0,
            step=1,
            value=int(st.session_state.get("score_team_b", 0)),
            key="score_team_b",
        )

    # global breakpoint slider
    slider_sidebar("Global breakpoint", key="global_breakpoint")

    st.sidebar.divider()

    render_team_block_sidebar("team_h", "Team H", team_params_df)

    st.sidebar.divider()

    render_team_block_sidebar("team_a", "Team 2", team_params_df)

    st.sidebar.divider()


# ------------------------------------------------------------
# Build config dataframes from current UI
# ------------------------------------------------------------
def build_global_df_from_ui() -> pd.DataFrame:
    # MUST be called global_breakpoint, because simulator looks for that
    return pd.DataFrame(
        [
            {
                "par_type": "global",
                "team_id": "global",
                "team_name": "global",
                "par_name": "global_breakpoint",
                "par_value": st.session_state.get("global_breakpoint", 0.0),
            }
        ]
    )


def build_team_df_from_ui(prefix: str) -> pd.DataFrame:
    """
    Build team df from current UI using the same names as in params_out_break_sideout.csv
    """
    team_id = st.session_state.get(f"{prefix}_current_team_id") or ""
    team_name = st.session_state.get(f"{prefix}_current_team_name") or ""

    rows = []

    # team-level breakpoint
    rows.append(
        {
            "par_type": "team",
            "team_id": team_id,
            "team_name": team_name,
            "par_name": "breakpoint_team_adjustment",
            "par_value": st.session_state.get(f"{prefix}_bp_adjustment", 0.0),
        }
    )
    # team-level sideout
    rows.append(
        {
            "par_type": "team",
            "team_id": team_id,
            "team_name": team_name,
            "par_name": "sideout_team_adjustment",
            "par_value": st.session_state.get(f"{prefix}_so_adjustment", 0.0),
        }
    )

    # rotation-level
    for pos in POSITIONS:
        rows.append(
            {
                "par_type": "team",
                "team_id": team_id,
                "team_name": team_name,
                "par_name": f"breakpoint_pos_{pos}",
                "par_value": st.session_state.get(
                    f"{prefix}_pos{pos}_bp_adjustment", 0.0
                ),
            }
        )
        rows.append(
            {
                "par_type": "team",
                "team_id": team_id,
                "team_name": team_name,
                "par_name": f"sideout_pos_{pos}",
                "par_value": st.session_state.get(
                    f"{prefix}_pos{pos}_so_adjustment", 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------
def compute_rotation_probability_matrix(
    global_df: pd.DataFrame,
    team_home_df: pd.DataFrame,
    team_away_df: pd.DataFrame,
    serve_team: str,
    score_team_a: int,
    score_team_b: int,
    is_tiebreak: bool,
):
    """
    Run the 6x6 grid using ONLY values from the UI.
    Then add the 0 rows/cols with averages.
    """
    from simulator import (
        VolleyballPointByPointSimulator,
        VolleyballProbabilitySimulator,
    )

    results = []

    for rot_h in range(1, 7):
        for rot_a in range(1, 7):
            base_sim = VolleyballPointByPointSimulator(seed=None)
            base_sim.load_parameters(
                global_df,
                team_home_df,
                team_away_df,
                match_type="Amichevole",
                match_date="08/10/2025",
            )

            if is_tiebreak:
                base_sim.set_initial_conditions(
                    set_won_h=2,
                    set_won_a=2,
                    point_won_h=score_team_a,
                    point_won_a=score_team_b,
                    p_h=rot_h,
                    p_a=rot_a,
                    serve_team=serve_team,
                    current_set=5,
                )
                base_sim.set_end_point(set_n=5, point_n=0)
            else:
                base_sim.set_initial_conditions(
                    set_won_h=0,
                    set_won_a=0,
                    point_won_h=score_team_a,
                    point_won_a=score_team_b,
                    p_h=rot_h,
                    p_a=rot_a,
                    serve_team=serve_team,
                    current_set=1,
                )
                base_sim.set_end_point(set_n=1, point_n=0)

            prob_sim = VolleyballProbabilitySimulator(base_sim)
            win_prob_h = prob_sim.home_win_analytical_calculations()

            results.append(
                {
                    "starting_rotation_h": rot_h,
                    "starting_rotation_a": rot_a,
                    "win_prob_h": win_prob_h,
                }
            )

    df_res = pd.DataFrame(results)

    # averages by home rotation
    home_avgs = (
        df_res.groupby("starting_rotation_h")["win_prob_h"]
        .mean()
        .reset_index()
    )
    home_avgs["starting_rotation_a"] = 0
    home_avgs = home_avgs[
        ["starting_rotation_h", "starting_rotation_a", "win_prob_h"]
    ]

    # averages by away rotation
    away_avgs = (
        df_res.groupby("starting_rotation_a")["win_prob_h"]
        .mean()
        .reset_index()
    )
    away_avgs["starting_rotation_h"] = 0
    away_avgs = away_avgs[
        ["starting_rotation_h", "starting_rotation_a", "win_prob_h"]
    ]

    total_averages = pd.DataFrame(
        index=[0],
        data=[[0, 0, away_avgs["win_prob_h"].mean()]],
        columns=["starting_rotation_h", "starting_rotation_a", "win_prob_h"],
    )

    df_res_all = pd.concat(
        [df_res, home_avgs, away_avgs, total_averages],
        ignore_index=True,
    )

    pivot = df_res_all.pivot(
        index="starting_rotation_h",
        columns="starting_rotation_a",
        values="win_prob_h",
    ).sort_index().sort_index(axis=1)

    return df_res_all, pivot


def style_rotation_matrix(pivot: pd.DataFrame):
    min_val = pivot.min().min()
    max_val = pivot.max().max()

    if pd.isna(min_val) or pd.isna(max_val):
        min_val, max_val = 0.0, 1.0

    diff = max_val - min_val

    if diff <= 0.005:
        def all_neutral(_):
            return "background-color: rgb(255, 255, 220)"

        return (
            pivot
            .style
            .format("{:.1%}")
            .map(all_neutral)
        )

    mid_val = min_val + diff / 2.0

    def val_to_color(v: float) -> str:
        if v <= mid_val:
            t = (v - min_val) / (mid_val - min_val)
            t = max(0.0, min(1.0, t))
            r = 255
            g = int(0 + (255 - 0) * t)
            b = int(0 + (255 - 0) * t)
        else:
            t = (v - mid_val) / (max_val - mid_val)
            t = max(0.0, min(1.0, t))
            r = int(255 + (0 - 255) * t)
            g = int(255 + (150 - 255) * t)
            b = int(255 + (0 - 255) * t)
        return f"background-color: rgb({r},{g},{b})"

    def color_cell(val):
        if pd.isna(val):
            return ""
        return val_to_color(float(val))

    return (
        pivot
        .style
        .format("{:.1%}")
        .map(color_cell)
    )


def run_simulation_and_store():
    global_df = build_global_df_from_ui()
    team_home_df = build_team_df_from_ui("team_h")
    team_away_df = build_team_df_from_ui("team_a")
    score_team_a = int(st.session_state.get("score_team_a", 0))
    score_team_b = int(st.session_state.get("score_team_b", 0))
    is_tiebreak = bool(st.session_state.get("tiebreak", False))

    home_label = (
        st.session_state.get("team_h_current_team_name")
        or st.session_state.get("team_h_current_team_id")
        or "home team"
    )
    away_label = (
        st.session_state.get("team_a_current_team_name")
        or st.session_state.get("team_a_current_team_id")
        or "away team"
    )

    results = {}
    for serve_team in ("h", "a"):
        df_res_all, pivot = compute_rotation_probability_matrix(
            global_df,
            team_home_df,
            team_away_df,
            serve_team,
            score_team_a,
            score_team_b,
            is_tiebreak,
        )
        results[serve_team] = {
            "df": df_res_all,
            "pivot": pivot,
        }

    st.session_state["last_rotation_results"] = results
    st.session_state["last_rotation_team_label_home"] = home_label
    st.session_state["last_rotation_team_label_away"] = away_label
    st.session_state["last_rotation_global_df"] = global_df
    st.session_state["last_rotation_team_home_df"] = team_home_df
    st.session_state["last_rotation_team_away_df"] = team_away_df
    st.session_state["last_rotation_score_team_a"] = score_team_a
    st.session_state["last_rotation_score_team_b"] = score_team_b
    st.session_state["last_rotation_is_tiebreak"] = is_tiebreak


def show_square_matrix(styled, pivot_df: pd.DataFrame):
    n_rows, n_cols = pivot_df.shape
    cell_w = 90
    cell_h = 38
    width = n_cols * cell_w
    height = n_rows * cell_h + 40
    st.dataframe(styled, width=width, height=height)


def prepare_pivot_for_display(pivot: pd.DataFrame, away_label: str) -> pd.DataFrame:
    display = pivot.rename(index={0: "AVG"}, columns={0: "AVG"})
    # Ensure all index/column labels are strings to avoid PyArrow mixed-type error
    display.index = display.index.astype(str)
    display.columns = display.columns.astype(str)

    display.index.name = "-"
    display.columns.name = f"starting rotation of {away_label}"
    return display


def _pivot_without_avg(pivot: pd.DataFrame) -> pd.DataFrame:
    return pivot.loc[pivot.index != "AVG", pivot.columns != "AVG"]


def best_away_response_table(
    pivot: pd.DataFrame, home_label: str, away_label: str
) -> Optional[pd.DataFrame]:
    cleaned = _pivot_without_avg(pivot)
    if cleaned.empty:
        return None

    def get_best_2(row):
        best_2 = row.nsmallest(2)
        if len(best_2) > 1:
            rot1 = best_2.index[0]
            prob1 = best_2.iloc[0]
            rot2 = best_2.index[1]
            prob2 = best_2.iloc[1]
            return f"{rot1} ({prob1:.1%}), {rot2} ({prob2:.1%})"
        elif len(best_2) == 1:
            rot1 = best_2.index[0]
            prob1 = best_2.iloc[0]
            return f"{rot1} ({prob1:.1%})"
        return ""

    best_away_series = cleaned.apply(get_best_2, axis=1)
    df = best_away_series.reset_index()
    df.columns = [
        f"{home_label} rotation",
        f"Best 2 {away_label} rotations",
    ]
    return df


def best_home_response_table(
    pivot: pd.DataFrame, home_label: str, away_label: str
) -> Optional[pd.DataFrame]:
    cleaned = _pivot_without_avg(pivot)
    if cleaned.empty:
        return None

    def get_best_2(col):
        best_2 = col.nlargest(2)
        if len(best_2) > 1:
            rot1 = best_2.index[0]
            prob1 = best_2.iloc[0]
            rot2 = best_2.index[1]
            prob2 = best_2.iloc[1]
            return f"{rot1} ({prob1:.1%}), {rot2} ({prob2:.1%})"
        elif len(best_2) == 1:
            rot1 = best_2.index[0]
            prob1 = best_2.iloc[0]
            return f"{rot1} ({prob1:.1%})"
        return ""

    best_home_series = cleaned.apply(get_best_2, axis=0)
    df = best_home_series.reset_index()
    df.columns = [
        f"{away_label} rotation",
        f"Best 2 {home_label} rotations",
    ]
    return df


def style_param_table(df: pd.DataFrame):
    if "par_value" not in df.columns:
        # If it's a pivot table, the values are the cells themselves
        # We can check if it looks like numeric data we want to color
        pass
    else:
        # Original long format
        return _style_long_param_table(df)
    
    # For pivot table (wide format)
    return _style_wide_param_table(df)

def _style_long_param_table(df: pd.DataFrame):
    def color_val(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return ""
        return _get_color_style(v)

    styled = df.style.map(color_val, subset=["par_value"])
    return styled

def _style_wide_param_table(df: pd.DataFrame):
    def color_val(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return ""
        return _get_color_style(v)
    
    return df.style.map(color_val)

def _get_color_style(v: float) -> str:
    if v <= -0.5:
        return "background-color: rgb(255, 0, 0)"
    if v >= 0.5:
        return "background-color: rgb(0, 150, 0)"

    if v < 0:
        t = (v + 0.5) / 0.5
        r = 255
        g = int(255 * t)
        b = int(255 * t)
    else:
        t = v / 0.5
        r = int(255 * (1 - t))
        g = int(255 - (255 - 150) * t)
        b = int(255 * (1 - t))
    return f"background-color: rgb({r},{g},{b})"


# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
def page_rotation_main():
    df_all = load_params()
    team_params_df = get_team_params(df_all)
    global_breakpoint_default = get_global_breakpoint_default(df_all)

    rotation_simulator_controls_in_sidebar(team_params_df, global_breakpoint_default)

    

    if "last_rotation_results" in st.session_state:
        home_label = st.session_state.get("last_rotation_team_label_home") or "home team"
        away_label = st.session_state.get("last_rotation_team_label_away") or "away team"
        st.title(f"Rotation simulator: Probability of Home team {home_label} winning")

        results = st.session_state["last_rotation_results"]
        col_h, col_a = st.columns(2)

        serve_to_label = [
            ("h", f"First Serve Home: {home_label}"),
            ("a", f"First Serve Away: {away_label}"),
        ]
        for (serve_team, label), col in zip(serve_to_label, (col_h, col_a)):
            with col:
                st.markdown(f"### {label}")
                st.caption(
                    f"Rows: starting rotation of {home_label}; columns: starting rotation "
                    f"of {away_label}"
                )
                pivot_df = results.get(serve_team, {}).get("pivot")
                if pivot_df is None:
                    st.info("No data yet.")
                    continue
                display_pivot = prepare_pivot_for_display(pivot_df, away_label)
                styled = style_rotation_matrix(display_pivot)
                show_square_matrix(styled, display_pivot)

                table_left, table_right = st.columns(2)

                best_away = best_away_response_table(pivot_df, home_label, away_label)
                with table_left:
                    if best_away is not None:
                        st.markdown(
                            f"**For each {home_label} rotation, 2 toughest replies from {away_label}**"
                        )
                        st.table(best_away.style.hide())

                best_home = best_home_response_table(pivot_df, home_label, away_label)
                with table_right:
                    if best_home is not None:
                        st.markdown(
                            f"**For each {away_label} rotation, 2 best answers from {home_label}**"
                        )
                        st.table(best_home.style.hide())

        with st.expander("All results (including 0 rows/cols)"):
            col_h_table, col_a_table = st.columns(2)
            with col_h_table:
                st.markdown(f"**Home {home_label}**")
                df_home = results.get("h", {}).get("df")
                if df_home is None:
                    st.info("Run the simulator to see data.")
                else:
                    st.dataframe(df_home)
            with col_a_table:
                st.markdown(f"**Away {away_label}**")
                df_away = results.get("a", {}).get("df")
                if df_away is None:
                    st.info("Run the simulator to see data.")
                else:
                    st.dataframe(df_away)

        with st.expander("Config sent to simulator"):
            st.markdown("**global_df**")
            global_df = st.session_state["last_rotation_global_df"]
            st.dataframe(style_param_table(global_df))

            team_home_df = st.session_state["last_rotation_team_home_df"]
            team_away_df = st.session_state["last_rotation_team_away_df"]

            c1, c2 = st.columns(2)

            # split home
            home_break_df = team_home_df[
                team_home_df["par_name"].str.startswith("breakpoint_")
            ].reset_index(drop=True)
            home_sideout_df = team_home_df[
                team_home_df["par_name"].str.startswith("sideout_")
            ].reset_index(drop=True)

            # split away
            away_break_df = team_away_df[
                team_away_df["par_name"].str.startswith("breakpoint_")
            ].reset_index(drop=True)
            away_sideout_df = team_away_df[
                team_away_df["par_name"].str.startswith("sideout_")
            ].reset_index(drop=True)

            with c1:
                st.markdown("**team_home_df – breakpoint params**")
                st.dataframe(style_param_table(home_break_df))

                st.markdown("**team_home_df – sideout params**")
                st.dataframe(style_param_table(home_sideout_df))

            with c2:
                st.markdown("**team_away_df – breakpoint params**")
                st.dataframe(style_param_table(away_break_df))

                st.markdown("**team_away_df – sideout params**")
                st.dataframe(style_param_table(away_sideout_df))

            st.markdown("**start scores / serve / tiebreak**")
            st.write(
                {
                    "home team": home_label,
                    "away team": away_label,
                    "score_team_a": st.session_state.get("last_rotation_score_team_a"),
                    "score_team_b": st.session_state.get("last_rotation_score_team_b"),
                    "serve_scenarios": ["home first serve", "away first serve"],
                    "tiebreak": st.session_state.get("last_rotation_is_tiebreak"),
                }
            )

    else:
        st.info("Set your parameters and click APPLY to run the rotation grid.")


def page_teams_summary():
    st.title("Teams Summary")
    
    df_all = load_params()
    team_params_df = get_team_params(df_all)
    
    # Pivot: rows=team_name, cols=par_name, values=par_value
    pivot = team_params_df.pivot(
        index="team_name",
        columns="par_name",
        values="par_value"
    )
    
    # Define column order
    bp_cols = ["breakpoint_team_adjustment"] + [f"breakpoint_pos_{i}" for i in range(1, 7)]
    so_cols = ["sideout_team_adjustment"] + [f"sideout_pos_{i}" for i in range(1, 7)]
    ordered_cols = bp_cols + so_cols
    
    # Filter to only existing columns (in case some are missing)
    existing_cols = [c for c in ordered_cols if c in pivot.columns]
    pivot = pivot[existing_cols]
    
    # Create MultiIndex columns for visual separation
    new_columns = []
    for col in pivot.columns:
        if col == "breakpoint_team_adjustment":
            new_columns.append(("Breakpoint", "Team"))
        elif col == "sideout_team_adjustment":
            new_columns.append(("Sideout", "Team"))
        elif col.startswith("breakpoint_pos_"):
            pos = col.split("_")[-1]
            new_columns.append(("Breakpoint", f"Pos{pos}"))
        elif col.startswith("sideout_pos_"):
            pos = col.split("_")[-1]
            new_columns.append(("Sideout", f"Pos{pos}"))
        else:
            # Fallback
            new_columns.append(("Other", col))
            
    pivot.columns = pd.MultiIndex.from_tuples(new_columns)
    
    # ------------------------------------------------------------------
    # Scatterplot
    # ------------------------------------------------------------------
    # Prepare data for plot (flatten MultiIndex)
    plot_df = pivot.copy()
    # Flatten columns: ("Breakpoint", "Team") -> "Breakpoint_Team"
    plot_df.columns = [f"{c[0]}_{c[1]}" for c in plot_df.columns]
    plot_df = plot_df.reset_index() # make team_name a column
    
    # Create scatterplot
    hover_cols = [c for c in plot_df.columns if "Pos" in c]
    # Format hover data to 2 decimal places
    hover_data_dict = {c: ":.2f" for c in hover_cols}
    hover_data_dict["Breakpoint_Team"] = ":.2f"
    hover_data_dict["Sideout_Team"] = ":.2f"

    fig = px.scatter(
        plot_df,
        x="Breakpoint_Team",
        y="Sideout_Team",
        hover_name="team_name",
        hover_data=hover_data_dict,
        title="Team Breakpoint vs Sideout Strength",
        width=600,
        height=600,
        labels={
            "Breakpoint_Team": "Breakpoint Strength (Team)",
            "Sideout_Team": "Sideout Strength (Team)"
        }
    )
    
    # Enforce square aspect ratio and ensure grid is visible
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey',
        dtick=0.1,
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey',
        dtick=0.1,
    )
    
    # Add a zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Display in half width
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig, width="stretch")
    
    # Style and display with 2 decimal formatting
    st.dataframe(
        style_param_table(pivot).format("{:.2f}"),
        width="stretch",
        height=500,
    )


def wip_page_main():
    st.title("🚧 Work in progress")
    st.write("This page is not ready yet.")


import threading

class BackgroundLoader:
    def __init__(self):
        self.data = None
        self.is_loading = False
        self.thread = None
        self.error = None
        self.progress_text = ""

    def start_loading(self, folder_ids):
        if not self.is_loading and self.data is None:
            self.is_loading = True
            self.progress_text = "Starting load..."
            self.thread = threading.Thread(target=self._load, args=(folder_ids,))
            self.thread.start()

    def update_progress(self, current, total, message):
        self.progress_text = f"{message}"

    def _load(self, folder_ids):
        try:
            # We need to ensure we don't access st.secrets directly if it's not thread-safe, 
            # but usually it is fine. Passing folder_ids explicitly helps.
            df = load_matches_from_drive(folder_ids, progress_callback=self.update_progress)
            self.data = df
        except Exception as e:
            self.error = str(e)
        finally:
            self.is_loading = False

@st.cache_resource
def get_loader():
    return BackgroundLoader()

def perform_load_async():
    """Start async loading if not already started."""
    loader = get_loader()
    
    # If already loaded or loading, do nothing
    if loader.data is not None or loader.is_loading:
        return

    # Get config
    folder_ids = []
    if "gdrive" in st.secrets:
        if "folder_ids" in st.secrets["gdrive"]:
            folder_ids = st.secrets["gdrive"]["folder_ids"]
        elif "folder_id" in st.secrets["gdrive"]:
            folder_ids = [st.secrets["gdrive"]["folder_id"]]
    
    if folder_ids:
        loader.start_loading(folder_ids)
    else:
        st.error("No Google Drive folder configured in secrets.")


def page_load_data():
    st.title("Load Data from Google Drive")
    
    loader = get_loader()

    # Status Indicator
    if loader.is_loading:
        st.info(f"⏳ Loading data... {loader.progress_text}")
        if st.button("Check Status"):
            st.rerun()
    elif loader.data is not None:
        st.success(f"✅ Data loaded ({len(loader.data)} rallies).")
        st.session_state["loaded_matches_df"] = loader.data
        
        # Refresh button
        if st.button("Refresh Data"):
            # Reset loader
            loader.data = None
            loader.is_loading = False
            loader.progress_text = ""
            perform_load_async()
            st.rerun()
            
    elif loader.error:
        st.error(f"Error loading data: {loader.error}")
        if st.button("Retry"):
            loader.error = None
            perform_load_async()
            st.rerun()
    else:
        st.warning("No data loaded.")
        if st.button("Start Loading"):
            perform_load_async()
            st.rerun()

    # 2. Display Matches (only if data is in session state)
    if "loaded_matches_df" in st.session_state and st.session_state["loaded_matches_df"] is not None:
        df = st.session_state["loaded_matches_df"]
        
        # Group by match to get summary
        matches = []
        for file_id, group in df.groupby("file_id"):
            # Assuming group is sorted by rally order
            last_row = group.iloc[-1]
            first_row = group.iloc[0]
            
            match_date = first_row.get("match_date")
            match_type = first_row.get("match_type")
            team_home = first_row.get("team_h")
            team_away = first_row.get("team_a")
            
            # Set score
            sets_h = last_row.get("post_set_won_h")
            sets_a = last_row.get("post_set_won_a")
            set_score = f"{sets_h}-{sets_a}"
            
            # Set scores detail
            set_scores_str = []
            for set_num, set_group in group.groupby("set_number"):
                last_set_row = set_group.iloc[-1]
                ph = last_set_row["post_point_won_h"]
                pa = last_set_row["post_point_won_a"]
                set_scores_str.append(f"{ph}-{pa}")
            
            full_score_str = f"{set_score} ({', '.join(set_scores_str)})"
            
            matches.append({
                "Date": match_date,
                "Type": match_type,
                "Home": team_home,
                "Away": team_away,
                "Score": full_score_str,
                "file_id": file_id,
                "file_name": first_row.get("file_name")
            })
            
        matches_df = pd.DataFrame(matches).sort_values(['Date']).reset_index(drop=True)
        
        # Display table
        st.dataframe(
            matches_df[["Date", "Type", "Home", "Away", "Score"]],
            width="stretch"
        )
        
        st.divider()
        st.markdown("### Download Data")
        
        # 1. Download All
        if st.button("Download ALL Matches (Merged CSV)"):
            with st.spinner("Processing ALL matches... this may take a while..."):
                try:
                    # We need to process all files. 
                    # We can iterate over unique file_ids in the dataframe
                    unique_files = matches_df[["file_id", "file_name"]].drop_duplicates()
                    
                    all_full_dfs = []
                    for _, row in unique_files.iterrows():
                        file_name = row['file_name']
                        file_id = row['file_id']
                        logging.info(f"Processing file for download: {file_name} (ID: {file_id})")
                        
                        try:
                            # Use bytes to avoid encoding issues
                            content = read_file_bytes(file_id)
                            if content:
                                df_full = process_dv_file_content(content, file_name)
                                all_full_dfs.append(df_full)
                            else:
                                logging.error(f"Failed to download content for file: {file_name}")
                        except Exception as e:
                            logging.error(f"Error processing file {file_name}: {e}")
                            # Continue to next file instead of failing everything
                            continue
                    
                    if all_full_dfs:
                        merged_df = pd.concat(all_full_dfs, ignore_index=True)
                        if "match_date" in merged_df.columns:
                            merged_df = merged_df.sort_values(by=["match_date"]).reset_index(drop=True)
                        csv_all = merged_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Click to Download MERGED CSV",
                            data=csv_all,
                            file_name="all_matches_full.csv",
                            mime="text/csv",
                            key="dl_all"
                        )
                    else:
                        st.warning("No data to download.")
                        
                except Exception as e:
                    st.error(f"Error processing all files: {e}")

        st.markdown("#### Download Single Match")
        
        # Dropdown options
        # Format: "YYYY-MM-DD | Home vs Away"
        match_options = {}
        for _, row in matches_df.iterrows():
            label = f"{row['Date']} | {row['Home']} vs {row['Away']}"
            match_options[label] = row['file_id']
            
        selected_label = st.selectbox("Select Match", options=list(match_options.keys()))
        
        if selected_label:
            selected_file_id = match_options[selected_label]
            # Find file name
            selected_file_name = matches_df[matches_df['file_id'] == selected_file_id].iloc[0]['file_name']
            
            btn_key = f"btn_dl_single_{selected_file_id}"
            
            if st.button("Prepare CSV for Selected Match", key=btn_key):
                with st.spinner("Processing match data..."):
                    try:
                        # Use bytes to avoid encoding issues
                        content = read_file_bytes(selected_file_id)
                        if content:
                            full_df = process_dv_file_content(content, selected_file_name)
                            csv = full_df.to_csv(index=False).encode('utf-8')
                            
                            st.session_state[f"csv_{selected_file_id}"] = csv
                            # No rerun needed if we just show the button below conditionally, 
                            # but rerun helps update state cleanly.
                            st.rerun()
                        else:
                            st.error("Failed to download file content.")
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
            
            if f"csv_{selected_file_id}" in st.session_state:
                st.download_button(
                    label="Download CSV",
                    data=st.session_state[f"csv_{selected_file_id}"],
                    file_name=f"{selected_file_name}_full.csv",
                    mime="text/csv",
                    key=f"dl_{selected_file_id}"
                )


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="Rotation App", layout="wide")

    # Start Async Load
    perform_load_async()
    
    # Check loader status for sidebar notification
    loader = get_loader()
    if loader.is_loading:
        st.sidebar.info(f"⏳ Loading data... {loader.progress_text}")
    elif loader.data is not None:
        # Ensure session state is synced
        if "loaded_matches_df" not in st.session_state:
            st.session_state["loaded_matches_df"] = loader.data
        st.sidebar.success(f"✅ Data loaded ({len(loader.data)} rallies)")

    st.sidebar.title("Menu")
    page = st.sidebar.selectbox(
        "Select page",
        options=[PAGE_ROTATION, PAGE_TEAMS_SUMMARY, PAGE_LOAD_DATA, PAGE_WIP],
        index=0,
    )

    if page == PAGE_ROTATION:
        page_rotation_main()
    elif page == PAGE_TEAMS_SUMMARY:
        page_teams_summary()
    elif page == PAGE_LOAD_DATA:
        page_load_data()
    else:
        wip_page_main()


if __name__ == "__main__":
    main()
