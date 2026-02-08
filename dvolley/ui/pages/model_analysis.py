import streamlit as st
import pandas as pd

from dvolley.domain.backtest_engine import (
    run_loocv_backtest,
    run_sequential_backtest,
    calculate_metrics,
    plot_calibration,
)
from dvolley.domain.model_logistic_rotation import LogisticRotationModelNoHome
from dvolley.domain.model_empirical import EmpiricalModel, GlobalMeanModel, SimpleEmpiricalModel
from dvolley.services.data_loader import load_data_from_db


def page_model_analysis():
    st.title("Model Analysis")

    st.sidebar.markdown("### Model Configuration")

    model_choice = st.sidebar.selectbox(
        "Select Model",
        options=[
            "logistic_rotation_alpha_0.1",
            "logistic_rotation_alpha_0.05",
            "logistic_rotation_alpha_0.01",
            "logistic_rotation_alpha_0.005",
            "logistic_rotation_alpha_0.001",
            "empirical_global_only",
            "empirical_team",
            "empirical_team_rotation",
        ],
        index=2,
    )

    backtest_choice = st.sidebar.selectbox(
        "Backtest Method",
        options=["LOO", "weekly_sequential"],
        help="LOO: Leave-One-Out Cross Validation. Weekly Sequential: Train on past, predict future.",
    )

    st.sidebar.info(
        "**LOO**: Trains on N-1 matches, tests on 1. Good for small data.\n"
        "**Weekly Sequential**: Simulates real-world scenario. Trains on past weeks, tests on current week."
    )

    col1, col2 = st.columns(2)
    with col1:
        apply_all = st.button("Apply All Models (Compare)", type="primary")
    with col2:
        apply_single = st.button(f"Apply '{model_choice}' Only")

    def get_model_instance(name):
        if name.startswith("logistic_rotation_alpha_"):
            alpha = float(name.split("_")[-1])
            return LogisticRotationModelNoHome(alpha=alpha)
        if name == "empirical_global_only":
            return GlobalMeanModel()
        if name == "empirical_team":
            return SimpleEmpiricalModel()
        if name == "empirical_team_rotation":
            return EmpiricalModel()
        return None

    def preprocess_data_for_models(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "team_id_h" in df.columns:
            df["team_id_h"] = df["team_id_h"].astype(str)
        if "team_id_a" in df.columns:
            df["team_id_a"] = df["team_id_a"].astype(str)

        required = ["serve_team", "point_won_team", "p_h", "p_a", "team_id_h", "team_id_a"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Rally data missing columns: {missing}")
            return pd.DataFrame()
        return df

    df_raw = load_data_from_db()
    if df_raw.empty:
        st.error("No rally data available. Please load data first.")
        return

    df = preprocess_data_for_models(df_raw)
    if df.empty:
        st.warning("No valid data found for analysis.")
        return

    if apply_all:
        st.markdown("### Model Comparison")

        models_to_run = [
            "logistic_rotation_alpha_0.1",
            "logistic_rotation_alpha_0.01",
            "logistic_rotation_alpha_0.005",
            "logistic_rotation_alpha_0.001",
            "empirical_global_only",
            "empirical_team",
            "empirical_team_rotation",
        ]

        results = []
        calibration_plots = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, m_name in enumerate(models_to_run):
            status_text.text(f"Running backtest for {m_name}...")
            model = get_model_instance(m_name)

            if backtest_choice == "LOO":
                y_true, y_pred = run_loocv_backtest(model, df)
            else:
                y_true, y_pred = run_sequential_backtest(model, df)

            metrics = calculate_metrics(y_true, y_pred)
            metrics["Model"] = m_name
            results.append(metrics)

            plot_img = plot_calibration(y_true, y_pred, m_name)
            calibration_plots[m_name] = plot_img

            progress_bar.progress((i + 1) / len(models_to_run))

        status_text.text("Done!")

        res_df = pd.DataFrame(results)
        st.dataframe(res_df.style.highlight_min(subset=["Log Loss", "Brier Score"], color="lightgreen"))

        st.markdown("### Calibration Plots")
        cols = st.columns(3)
        for i, m_name in enumerate(models_to_run):
            with cols[i % 3]:
                st.image(calibration_plots[m_name], use_column_width=True)

    if apply_single or apply_all:
        st.divider()
        st.markdown(f"### Active Model: **{model_choice}**")

        with st.spinner(f"Fitting {model_choice} on ALL data..."):
            model = get_model_instance(model_choice)
            model.fit(df)

            sim_params = model.get_simulator_params()

            st.session_state["active_model"] = {
                "name": model_choice,
                "params": sim_params,
                "type": sim_params["type"],
            }

            st.success(f"Model {model_choice} fitted and set as ACTIVE.")
            st.info("This model will now be used in 'Teams Summary' and 'Rotation Simulator'.")

            if apply_single:
                st.markdown("#### Backtest Results for Selected Model")
                if backtest_choice == "LOO":
                    y_true, y_pred = run_loocv_backtest(model, df)
                else:
                    y_true, y_pred = run_sequential_backtest(model, df)

                metrics = calculate_metrics(y_true, y_pred)
                st.write(metrics)

                plot_img = plot_calibration(y_true, y_pred, model_choice)
                st.image(plot_img)
