import streamlit as st
import pandas as pd
from typing import Optional

# ------------------------------------------------------------
# App configuration / constants
# ------------------------------------------------------------
PAGE_ROTATION = "rotation_simulator"
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
    display.index.name = "-"
    display.columns.name = f"starting rotation of {away_label}"
    return display


def _pivot_without_avg(pivot: pd.DataFrame) -> pd.DataFrame:
    return pivot.loc[pivot.index != 0, pivot.columns != 0]


def best_away_response_table(
    pivot: pd.DataFrame, home_label: str, away_label: str
) -> Optional[pd.DataFrame]:
    cleaned = _pivot_without_avg(pivot)
    if cleaned.empty:
        return None
    best_away = cleaned.idxmin(axis=1).astype(int)
    df = best_away.reset_index()
    df.columns = [
        f"{home_label} rotation",
        f"Best {away_label} rotation",
    ]
    return df


def best_home_response_table(
    pivot: pd.DataFrame, home_label: str, away_label: str
) -> Optional[pd.DataFrame]:
    cleaned = _pivot_without_avg(pivot)
    if cleaned.empty:
        return None
    best_home = cleaned.idxmax(axis=0).astype(int)
    df = best_home.reset_index()
    df.columns = [
        f"{away_label} rotation",
        f"Best {home_label} rotation",
    ]
    return df


def style_param_table(df: pd.DataFrame):
    if "par_value" not in df.columns:
        return df.style

    def color_val(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return ""
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

    styled = df.style.map(color_val, subset=["par_value"])
    return styled


# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
def page_rotation_main():
    df_all = load_params()
    team_params_df = get_team_params(df_all)
    global_breakpoint_default = get_global_breakpoint_default(df_all)

    rotation_simulator_controls_in_sidebar(team_params_df, global_breakpoint_default)

    st.title("Rotation simulator")

    if "last_rotation_results" in st.session_state:
        home_label = st.session_state.get("last_rotation_team_label_home") or "home team"
        away_label = st.session_state.get("last_rotation_team_label_away") or "away team"

        st.subheader(f"Probability of Home team {home_label} winning")
        
        results = st.session_state["last_rotation_results"]
        col_h, col_a = st.columns(2)

        serve_to_label = [
            ("h", f"First Serve Home: {home_label}"),
            ("a", f"First Serve Away: {away_label}"),
        ]
        for (serve_team, label), col in zip(serve_to_label, (col_h, col_a)):
            with col:
                st.markdown(f"## {label}")
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
                            f"**For each {home_label} rotation, toughest reply from {away_label}**"
                        )
                        st.table(best_away.style.hide())

                best_home = best_home_response_table(pivot_df, home_label, away_label)
                with table_right:
                    if best_home is not None:
                        st.markdown(
                            f"**For each {away_label} rotation, best answer from {home_label}**"
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


def wip_page_main():
    st.title("🚧 Work in progress")
    st.write("This page is not ready yet.")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="Rotation App", layout="wide")

    st.sidebar.title("Menu")
    page = st.sidebar.selectbox(
        "Select page",
        options=[PAGE_ROTATION, PAGE_WIP],
        index=0,
    )

    if page == PAGE_ROTATION:
        page_rotation_main()
    else:
        wip_page_main()


if __name__ == "__main__":
    main()
