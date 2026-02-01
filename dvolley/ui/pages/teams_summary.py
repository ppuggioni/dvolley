import streamlit as st
import pandas as pd
import numpy as np

from .rotation import get_fitted_params_df, get_team_params, style_param_table


def page_teams_summary(loader, last_match_date: str | None = None):
    st.title("Teams Summary")

    active_model = st.session_state.get("active_model")

    if active_model and active_model["type"] == "empirical":
        st.info(f"Using Active Model: **{active_model['name']}** (Empirical)")

        params = active_model["params"]
        level = params.get("level", "global")
        global_mean = params.get("global_mean", 0.5)

        data = []

        if level == "team":
            break_team = params.get("break_team", {})
            sideout_team = params.get("sideout_team", {})
            all_teams = set(break_team.keys()) | set(sideout_team.keys())

            for t in all_teams:
                data.append(
                    {
                        "team_id": t,
                        "team_name": t,
                        "Breakpoint_Prob": break_team.get(t, global_mean),
                        "Sideout_Prob": sideout_team.get(t, global_mean),
                    }
                )

        elif level == "rotation":
            break_pos = params.get("break_pos", {})
            sideout_pos = params.get("sideout_pos", {})

            team_stats = {}
            for (tid, pos), val in break_pos.items():
                if tid not in team_stats:
                    team_stats[tid] = {"bp": [], "so": []}
                team_stats[tid]["bp"].append(val)

            for (tid, pos), val in sideout_pos.items():
                if tid not in team_stats:
                    team_stats[tid] = {"bp": [], "so": []}
                team_stats[tid]["so"].append(val)

            for tid, stats in team_stats.items():
                bp_avg = np.mean(stats["bp"]) if stats["bp"] else global_mean
                so_avg = np.mean(stats["so"]) if stats["so"] else global_mean
                data.append(
                    {
                        "team_id": tid,
                        "team_name": tid,
                        "Breakpoint_Prob": bp_avg,
                        "Sideout_Prob": so_avg,
                    }
                )
        else:
            st.warning("Global model selected. All teams have same probability.")
            st.write(f"Global Mean Probability: {global_mean:.4f}")
            return

        df = pd.DataFrame(data)

        if df.empty:
            st.warning("No data available for plotting.")
            return

        import plotly.express as px

        fig = px.scatter(
            df,
            x="Breakpoint_Prob",
            y="Sideout_Prob",
            hover_name="team_name",
            title="Team Average Breakpoint vs Sideout Probability (Empirical)",
            width=600,
            height=600,
        )

        fig.add_hline(y=global_mean, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=global_mean, line_dash="dash", line_color="gray", opacity=0.5)

        st.plotly_chart(fig)
        st.dataframe(df)
        return

    fitted_params_df = get_fitted_params_df()
    if fitted_params_df is None:
        if last_match_date:
            st.info(f"Last match date in DB: {last_match_date}")
        st.warning("No parameters fitted yet. Go to **Setup & Status** to load data and fit the model.")
        return

    team_params_df = get_team_params(fitted_params_df)

    pivot = team_params_df.pivot(
        index="team_name",
        columns="par_name",
        values="par_value",
    )

    bp_cols = ["breakpoint_team_adjustment"] + [f"breakpoint_pos_{i}" for i in range(1, 7)]
    so_cols = ["sideout_team_adjustment"] + [f"sideout_pos_{i}" for i in range(1, 7)]
    ordered_cols = bp_cols + so_cols

    existing_cols = [c for c in ordered_cols if c in pivot.columns]
    pivot = pivot[existing_cols]

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
            new_columns.append(("Other", col))

    pivot.columns = pd.MultiIndex.from_tuples(new_columns)

    plot_df = pivot.copy()
    plot_df.columns = [f"{c[0]}_{c[1]}" for c in plot_df.columns]
    plot_df = plot_df.reset_index()

    import plotly.express as px

    hover_cols = [c for c in plot_df.columns if "Pos" in c]
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
            "Sideout_Team": "Sideout Strength (Team)",
        },
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        dtick=0.1,
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        dtick=0.1,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    c1, _ = st.columns(2)
    with c1:
        st.plotly_chart(fig, width="stretch")

    styled_pivot = style_param_table(pivot)
    styled_pivot = styled_pivot.format("{:.2f}")
    st.dataframe(
        styled_pivot,
        width="stretch",
        height=500,
    )
