import time
import streamlit as st
import pandas as pd

from dvolley.config import DEFAULT_ALPHA
from dvolley.domain.model_logistic_rotation import LogisticRotationModelNoHome
from dvolley.domain.model_empirical import EmpiricalModel, GlobalMeanModel, SimpleEmpiricalModel
from dvolley.services.data_loader import (
    update_database,
    update_database_full,
    load_full_data_from_db,
    load_match_data_from_db,
)
from dvolley.ui.pages.rotation import refit_model_on_current_data


def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def _get_model_instance(name: str):
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


def _prep_model_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["team_id_h", "team_id_a"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    required = ["serve_team", "point_won_team", "p_h", "p_a", "team_id_h", "team_id_a"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Rally data missing columns: {missing}")
        return pd.DataFrame()
    return df


def fit_selected_model(
    active_df: pd.DataFrame,
    selected_model: str,
    *,
    show_messages: bool = True,
) -> bool:
    model_df = _prep_model_df(active_df)
    if model_df.empty:
        if show_messages:
            st.warning("No valid data available for fitting.")
        return False

    model = _get_model_instance(selected_model)
    if model is None:
        if show_messages:
            st.error(f"Unknown model: {selected_model}")
        return False

    model.fit(model_df)
    sim_params = model.get_simulator_params()
    st.session_state["active_model"] = {
        "name": selected_model,
        "params": sim_params,
        "type": sim_params.get("type"),
    }

    if sim_params.get("type") == "logistic":
        params_df = sim_params.get("params")
        if params_df is not None:
            st.session_state["fitted_params_df"] = params_df
            if show_messages:
                num_teams = len(
                    params_df[params_df["par_type"] == "team"]["team_id"].unique()
                )
                num_params = len(params_df)
                st.success("✅ Model refitted successfully!")
                st.info(
                    f"Fitted {num_params} parameters for {num_teams} teams "
                    f"using {len(active_df)} rallies."
                )
        else:
            if show_messages:
                st.warning("Model fit succeeded but no parameters were returned.")
    else:
        st.session_state.pop("fitted_params_df", None)
        if show_messages:
            st.success("✅ Model refitted successfully!")

    if show_messages:
        st.info("You can now use the Rotation Simulator and Teams Summary pages.")
    return True


def page_setup_status(loader, last_match_date: str | None = None):
    st.title("Setup & Status")

    perform_load_async = st.session_state.get("perform_load_async")
    if loader.data is None and not loader.is_loading and perform_load_async:
        perform_load_async()

    active_df = loader.data
    if active_df is None and "loaded_matches_df" in st.session_state:
        active_df = st.session_state["loaded_matches_df"]

    rally_count = len(active_df) if active_df is not None else 0
    fitted = "fitted_params_df" in st.session_state

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("DB connection", "OK")
    with col2:
        st.metric("Rallies loaded", rally_count)
    with col3:
        st.metric("Model fitted", "Yes" if fitted else "No")

    if last_match_date:
        st.caption(f"Last match date in DB: {last_match_date}")

    st.divider()

    col_sync, col_load, col_fit = st.columns(3, gap="large")

    with col_sync:
        st.markdown("### Sync from Google Drive")
        st.caption("Pull new `.dvw` files and upload to Supabase.")
        if st.button("Sync Google Drive â†’ DB", type="primary"):
            folder_ids = []
            if "gdrive" in st.secrets:
                if "folder_ids" in st.secrets["gdrive"]:
                    folder_ids = st.secrets["gdrive"]["folder_ids"]
                elif "folder_id" in st.secrets["gdrive"]:
                    folder_ids = [st.secrets["gdrive"]["folder_id"]]

            if not folder_ids:
                st.error("No Google Drive folder configured in secrets.")
            else:
                with st.status("Updating Database...", expanded=True) as status:
                    st.write("Checking for new Rally Data...")
                    try:
                        new_rallies = update_database(folder_ids)
                        if new_rallies:
                            st.success(
                                f"Uploaded {len(new_rallies)} new files to Rally Data: "
                                f"{', '.join(new_rallies)}"
                            )
                        else:
                            st.info("Rally Data is up to date.")
                    except Exception as e:
                        st.error(f"Error updating Rally Data: {e}")

                    st.write("Checking for new Touch Data...")
                    try:
                        new_touches = update_database_full(folder_ids)
                        if new_touches:
                            st.success(
                                f"Uploaded {len(new_touches)} new matches to Touch Data: "
                                f"{', '.join(new_touches)}"
                            )
                        else:
                            st.info("Touch Data is up to date.")
                    except Exception as e:
                        st.error(f"Error updating Touch Data: {e}")

                    status.update(label="Database Update Complete", state="complete", expanded=False)

                st.cache_data.clear()
                loader.data = None
                perform_load_async = st.session_state.get("perform_load_async")
                if perform_load_async:
                    perform_load_async()

    with col_load:
        st.markdown("### Load DB into the app")
        if loader.is_loading:
            st.info(f"â³ Loading data from Database... {loader.progress_text}")
            time.sleep(0.5)
            _safe_rerun()
        elif loader.error:
            st.error(f"Load failed: {loader.error}")
        elif active_df is not None:
            st.success(f"âœ… Data loaded ({len(active_df)} rallies)")
        if st.button("Reload data from DB"):
            loader.data = None
            loader.is_loading = False
            loader.progress_text = ""
            if perform_load_async:
                perform_load_async()
            _safe_rerun()

    with col_fit:
        st.markdown("### Fit model")
        model_names = [
            "logistic_rotation_alpha_0.1",
            "logistic_rotation_alpha_0.05",
            "logistic_rotation_alpha_0.01",
            "logistic_rotation_alpha_0.005",
            "logistic_rotation_alpha_0.001",
            "empirical_global_only",
            "empirical_team",
            "empirical_team_rotation",
        ]
        default_index = model_names.index("logistic_rotation_alpha_0.005")
        selected_model = st.selectbox(
            "Model",
            options=model_names,
            index=default_index,
            key="fit_model_option",
        )
        if active_df is None or active_df.empty:
            st.info("Load data first to fit the model.")
        else:
            if st.button(f"Fit model ({selected_model})"):
                with st.spinner("Refitting model... this may take a moment..."):
                    fit_selected_model(active_df, selected_model, show_messages=True)

    if active_df is not None and not active_df.empty:
        st.divider()
        st.markdown("### Matches in Database")

        matches = []
        for file_id, group in active_df.groupby("file_id"):
            last_row = group.iloc[-1]
            first_row = group.iloc[0]

            match_date = first_row.get("match_date")
            match_type = first_row.get("match_type")
            team_home = first_row.get("team_h")
            team_away = first_row.get("team_a")
            match_id = first_row.get("match_alternative_id")

            sets_h = last_row.get("post_set_won_h")
            sets_a = last_row.get("post_set_won_a")
            set_score = f"{sets_h}-{sets_a}"

            set_scores_str = []
            for _, set_group in group.groupby("set_number"):
                last_set_row = set_group.iloc[-1]
                ph = last_set_row["post_point_won_h"]
                pa = last_set_row["post_point_won_a"]
                set_scores_str.append(f"{ph}-{pa}")

            full_score_str = f"{set_score} ({', '.join(set_scores_str)})"

            matches.append(
                {
                    "Match ID": match_id,
                    "Date": match_date,
                    "Type": match_type,
                    "Home": team_home,
                    "Away": team_away,
                    "Score": full_score_str,
                }
            )

        matches_df = pd.DataFrame(matches).sort_values(["Date"]).reset_index(drop=True)
        if "Match ID" in matches_df.columns:
            matches_df = matches_df.set_index("Match ID", drop=True)

        table_df = matches_df.copy()
        table_df.insert(0, "Select", False)
        edited_table = st.data_editor(
            table_df[["Select", "Date", "Type", "Home", "Away", "Score"]],
            disabled=["Date", "Type", "Home", "Away", "Score"],
            hide_index=True,
            use_container_width=True,
            key="matches_table_select",
        )

        selected_match_ids = edited_table[edited_table["Select"]].index.tolist()
        selected_match_id = selected_match_ids[0] if selected_match_ids else None
        if len(selected_match_ids) > 1:
            st.warning("Select only one match for single-match downloads.")

        st.divider()
        st.markdown("### Downloads")

        st.markdown("#### Rally-level data")
        st.caption("All rallies for all matches (may take a while).")
        if st.button("Download ALL Rally Data"):
            csv_all = active_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Click to download rally_all.csv",
                data=csv_all,
                file_name="rally_all.csv",
                mime="text/csv",
                key="dl_rally_all",
            )

        if selected_match_id:
            if st.button("Download selected rally match"):
                match_df = active_df[active_df["match_alternative_id"] == selected_match_id]
                if not match_df.empty:
                    csv_match = match_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Click to download rally_match.csv",
                        data=csv_match,
                        file_name=f"rally_match_{selected_match_id}.csv",
                        mime="text/csv",
                        key="dl_rally_match",
                    )
                else:
                    st.warning(f"No rally data found for match ID: {selected_match_id}")
        else:
            st.info("Select a match in the table above to download a single rally match.")

        st.markdown("#### Point-by-point (touch-level) data")
        st.caption("Full touch-level dataset (may take a while).")
        if st.button("Download ALL Point-by-Point Data"):
            with st.spinner("Fetching full touch-level data..."):
                full_df = load_full_data_from_db()
                if not full_df.empty:
                    csv_all = full_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Click to download touch_all.csv",
                        data=csv_all,
                        file_name="touch_all.csv",
                        mime="text/csv",
                        key="dl_touch_all",
                    )
                else:
                    st.warning("No touch-level data found in database.")

        if selected_match_id:
            if st.button("Download selected point-by-point match"):
                with st.spinner("Fetching match data..."):
                    match_df = load_match_data_from_db(selected_match_id)
                    if not match_df.empty:
                        csv_match = match_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Click to download touch_match.csv",
                            data=csv_match,
                            file_name=f"touch_match_{selected_match_id}.csv",
                            mime="text/csv",
                            key="dl_touch_match",
                        )
                    else:
                        st.warning(f"No touch-level data found for match ID: {selected_match_id}")
        else:
            st.info("Select a match in the table above to download a single point-by-point match.")
