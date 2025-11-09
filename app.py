import streamlit as st
import pandas as pd

# ------------------------------------------------------------
# App configuration / constants
# ------------------------------------------------------------
PAGE_ROTATION = "rotation_simulator"
PAGE_WIP = "work in progress"

POSITIONS = range(1, 7)
SLIDER_MIN = -2.0
SLIDER_MAX = 2.0
SLIDER_DEFAULT = 0.0
SLIDER_STEP = 0.01


# ------------------------------------------------------------
# Load CSV once
# ------------------------------------------------------------
@st.cache_data
def load_params(path: str = "./params_out.csv") -> pd.DataFrame:
    """Load the whole params file (global + team)."""
    return pd.read_csv(path, dtype={"team_id": str})


def get_team_params(df_all: pd.DataFrame) -> pd.DataFrame:
    return df_all[df_all["par_type"] == "team"].copy()


def get_global_serve_default(df_all: pd.DataFrame) -> float:
    m = (df_all["par_type"] == "global") & (df_all["par_name"] == "global_serve")
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
    # we won't set current_team_id/name here, we do it in the caller


def apply_team_preset_if_changed(
        prefix: str,
        selected_team_id: str | None,
        team_params_df: pd.DataFrame,
):
    prev_key = f"{prefix}_selected_team_id_prev"
    prev_val = st.session_state.get(prev_key)

    # if it's empty/Reset, caller will handle
    if not selected_team_id or selected_team_id == "Reset":
        st.session_state[prev_key] = selected_team_id
        return

    if selected_team_id != prev_val:
        rows = team_params_df[team_params_df["team_id"] == selected_team_id]
        for _, row in rows.iterrows():
            par_name = row["par_name"]
            par_value = float(row["par_value"])

            if par_name == "serve_team_adjustment":
                st.session_state[f"{prefix}_bp_adjustment"] = par_value
            elif par_name.startswith("serve_pos_"):
                pos = par_name.split("_")[-1]
                st.session_state[f"{prefix}_pos{pos}_bp_adjustment"] = par_value
            elif par_name == "receive_team_adjustment":
                st.session_state[f"{prefix}_so_adjustment"] = par_value
            elif par_name.startswith("receive_pos_"):
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
        # only do the zeroing when user ACTUALLY switched to Reset now
        if prev_sel != "Reset":
            reset_team_sliders(team_prefix)
        # treat Reset as a fake team we can still tweak
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

    # BP
    sb_left.markdown("**BP adjustments**")
    slider_sidebar(
        f"{team_name} BP (global)",
        key=f"{team_prefix}_bp_adjustment",
        component=sb_left,
    )
    for pos in POSITIONS:
        slider_sidebar(
            f"{team_name} pos{pos} BP",
            key=f"{team_prefix}_pos{pos}_bp_adjustment",
            component=sb_left,
        )

    # SO
    sb_right.markdown("**SO adjustments**")
    slider_sidebar(
        f"{team_name} SO (global)",
        key=f"{team_prefix}_so_adjustment",
        component=sb_right,
    )
    for pos in POSITIONS:
        slider_sidebar(
            f"{team_name} pos{pos} SO",
            key=f"{team_prefix}_pos{pos}_so_adjustment",
            component=sb_right,
        )


def rotation_simulator_controls_in_sidebar(team_params_df: pd.DataFrame, global_serve_default: float):
    """
    Sidebar controls with APPLY at the top.
    """
    st.sidebar.markdown("## Rotation simulator")

    # APPLY at the top
    if st.sidebar.button("APPLY", type="primary", use_container_width=True):
        run_simulation_and_store()

    # seed global_serve once
    if "global_serve" not in st.session_state:
        st.session_state["global_serve"] = global_serve_default

    # tiebreak toggle
    st.sidebar.checkbox("Tiebreak", key="tiebreak")

    # first serve
    st.sidebar.pills(
        label="First serve",
        options=["h", "a"],
        default="h",
        selection_mode="single",
        key="first_serve",
    )

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

    # global serve slider
    slider_sidebar("Global serve", key="global_serve")

    st.sidebar.divider()

    render_team_block_sidebar("team_h", "Team H", team_params_df)

    st.sidebar.divider()

    render_team_block_sidebar("team_a", "Team 2", team_params_df)

    st.sidebar.divider()


# ------------------------------------------------------------
# Build fake config dataframes from current UI
# ------------------------------------------------------------
def build_global_df_from_ui() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "par_type": "global",
                "team_id": "global",
                "team_name": "global",
                "par_name": "global_serve",
                "par_value": st.session_state.get("global_serve", 0.0),
            }
        ]
    )


def build_team_df_from_ui(prefix: str) -> pd.DataFrame:
    """
    Build team df from current UI.
    If user chose 'Reset', current_team_id/name will be 'Reset', which is fine.
    """
    team_id = st.session_state.get(f"{prefix}_current_team_id") or ""
    team_name = st.session_state.get(f"{prefix}_current_team_name") or ""

    rows = []

    rows.append(
        {
            "par_type": "team",
            "team_id": team_id,
            "team_name": team_name,
            "par_name": "serve_team_adjustment",
            "par_value": st.session_state.get(f"{prefix}_bp_adjustment", 0.0),
        }
    )
    rows.append(
        {
            "par_type": "team",
            "team_id": team_id,
            "team_name": team_name,
            "par_name": "receive_team_adjustment",
            "par_value": st.session_state.get(f"{prefix}_so_adjustment", 0.0),
        }
    )

    for pos in POSITIONS:
        rows.append(
            {
                "par_type": "team",
                "team_id": team_id,
                "team_name": team_name,
                "par_name": f"serve_pos_{pos}",
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
                "par_name": f"receive_pos_{pos}",
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
    """
    - show values as % with 1 decimal
    - color scale: min -> red, middle -> white, max -> green
    - scale is based on actual min/max in the matrix
    - if max-min <= 0.005 (0.5 percentage points) paint everything neutral
    """
    # actual min/max over the matrix
    min_val = pivot.min().min()
    max_val = pivot.max().max()

    if pd.isna(min_val) or pd.isna(max_val):
        min_val, max_val = 0.0, 1.0

    diff = max_val - min_val

    # if everything is basically the same
    if diff <= 0.005:
        def all_neutral(_):
            return "background-color: rgb(255, 255, 220)"

        return (
            pivot
            .style
            .format("{:.1%}")
            .applymap(all_neutral)
        )

    mid_val = min_val + diff / 2.0

    def val_to_color(v: float) -> str:
        # red -> white for [min .. mid]
        # white -> green for [mid .. max]

        if v <= mid_val:
            # t in [0..1] from min to mid
            t = (v - min_val) / (mid_val - min_val)
            t = max(0.0, min(1.0, t))
            # red (255,0,0) to white (255,255,255)
            r = 255
            g = int(0 + (255 - 0) * t)
            b = int(0 + (255 - 0) * t)
        else:
            # t in [0..1] from mid to max
            t = (v - mid_val) / (max_val - mid_val)
            t = max(0.0, min(1.0, t))
            # white (255,255,255) to green (0,150,0)
            r = int(255 + (0 - 255) * t)  # 255 -> 0
            g = int(255 + (150 - 255) * t)  # 255 -> 150
            b = int(255 + (0 - 255) * t)  # 255 -> 0

        return f"background-color: rgb({r},{g},{b})"

    def color_cell(val):
        if pd.isna(val):
            return ""
        return val_to_color(float(val))

    return (
        pivot
        .style
        .format("{:.1%}")
        .applymap(color_cell)
    )

    def color_cell(val):
        if pd.isna(val):
            return ""
        return val_to_color(float(val))

    return pivot.style.format("{:.3f}").applymap(color_cell)

    def color_cell(val):
        if pd.isna(val):
            return ""
        return val_to_color(float(val))

    return pivot.style.format("{:.3f}").applymap(color_cell)


def run_simulation_and_store():
    """
    Called ONLY when user clicks APPLY.
    """
    global_df = build_global_df_from_ui()
    team_home_df = build_team_df_from_ui("team_h")
    team_away_df = build_team_df_from_ui("team_a")
    serve_team = st.session_state.get("first_serve", "h")
    score_team_a = int(st.session_state.get("score_team_a", 0))
    score_team_b = int(st.session_state.get("score_team_b", 0))
    is_tiebreak = bool(st.session_state.get("tiebreak", False))

    df_res_all, pivot = compute_rotation_probability_matrix(
        global_df,
        team_home_df,
        team_away_df,
        serve_team,
        score_team_a,
        score_team_b,
        is_tiebreak,
    )

    st.session_state["last_rotation_df"] = df_res_all
    st.session_state["last_rotation_pivot"] = pivot
    st.session_state["last_rotation_global_df"] = global_df
    st.session_state["last_rotation_team_home_df"] = team_home_df
    st.session_state["last_rotation_team_away_df"] = team_away_df
    st.session_state["last_rotation_score_team_a"] = score_team_a
    st.session_state["last_rotation_score_team_b"] = score_team_b
    st.session_state["last_rotation_serve_team"] = serve_team
    st.session_state["last_rotation_is_tiebreak"] = is_tiebreak


def show_square_matrix(styled, pivot_df: pd.DataFrame):
    """
    Render the styled pivot with width/height tuned so cells look squarish.
    `styled` is the Styler returned by your style_rotation_matrix.
    """
    n_rows, n_cols = pivot_df.shape

    # tweak to taste
    cell_w = 90  # px per column
    cell_h = 38  # px per row (including header height-ish)

    width = n_cols * cell_w
    height = n_rows * cell_h + 40  # little extra for headers

    st.dataframe(styled, use_container_width=False, width=width, height=height)


# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
def page_rotation_main():
    df_all = load_params()
    team_params_df = get_team_params(df_all)
    global_serve_default = get_global_serve_default(df_all)

    rotation_simulator_controls_in_sidebar(team_params_df, global_serve_default)

    st.title("Rotation simulator")

    if "last_rotation_pivot" in st.session_state:
        st.subheader("rotation_probability_matrix (home rows, away cols)")
        pivot_df = st.session_state["last_rotation_pivot"]
        styled = style_rotation_matrix(pivot_df)
        show_square_matrix(styled, pivot_df)

        with st.expander("All results (including 0 rows/cols)"):
            st.dataframe(st.session_state["last_rotation_df"])

        with st.expander("Config sent to simulator"):
            st.markdown("**global_df**")
            st.dataframe(st.session_state["last_rotation_global_df"])
            st.markdown("**team_home_df**")
            st.dataframe(st.session_state["last_rotation_team_home_df"])
            st.markdown("**team_away_df**")
            st.dataframe(st.session_state["last_rotation_team_away_df"])
            st.markdown("**start scores / serve / tiebreak**")
            st.write(
                {
                    "score_team_a": st.session_state.get("last_rotation_score_team_a"),
                    "score_team_b": st.session_state.get("last_rotation_score_team_b"),
                    "serve_team": st.session_state.get("last_rotation_serve_team"),
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
