import streamlit as st
import pandas as pd

from dvolley.services.data_loader import (
    load_full_data_from_db,
    load_match_data_from_db,
    update_database_full,
    update_database,
)
from dvolley.ui.pages.rotation import refit_model_on_current_data
from dvolley.config import DEFAULT_ALPHA


def page_load_data(loader, last_match_date: str | None = None):
    st.title("Data Management")

    st.markdown("### 1. Update Database from Google Drive")
    st.info("This will scan the configured Google Drive folders for new `.dvw` files and upload them to the Supabase database.")

    if st.button("Update Database Now", type="primary"):
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
                        st.success(f"Uploaded {len(new_rallies)} new files to Rally Data: {', '.join(new_rallies)}")
                    else:
                        st.info("Rally Data is up to date.")
                except Exception as e:
                    st.error(f"Error updating Rally Data: {e}")

                st.write("Checking for new Touch Data...")
                try:
                    new_touches = update_database_full(folder_ids)
                    if new_touches:
                        st.success(f"Uploaded {len(new_touches)} new matches to Touch Data: {', '.join(new_touches)}")
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

    st.divider()

    st.markdown("### 2. Application Data Status")
    if loader.is_loading:
        st.info(f"⏳ Loading data from Database... {loader.progress_text}")
        if st.button("Check Status"):
            st.rerun()
    elif loader.data is not None:
        st.success(f"✅ Data loaded from Database ({len(loader.data)} rallies).")
        st.session_state["loaded_matches_df"] = loader.data

        if st.button("Reload Data from DB"):
            loader.data = None
            loader.is_loading = False
            loader.progress_text = ""
            perform_load_async = st.session_state.get("perform_load_async")
            if perform_load_async:
                perform_load_async()
            st.rerun()
    elif loader.error:
        st.error(f"Error loading data: {loader.error}")
        if st.button("Retry"):
            loader.error = None
            perform_load_async = st.session_state.get("perform_load_async")
            if perform_load_async:
                perform_load_async()
            st.rerun()
    else:
        st.warning("No data loaded.")
        if st.button("Start Loading"):
            perform_load_async = st.session_state.get("perform_load_async")
            if perform_load_async:
                perform_load_async()
            st.rerun()

    if "loaded_matches_df" in st.session_state and st.session_state["loaded_matches_df"] is not None:
        df = st.session_state["loaded_matches_df"]

        matches = []
        for file_id, group in df.groupby("file_id"):
            last_row = group.iloc[-1]
            first_row = group.iloc[0]

            match_date = first_row.get("match_date")
            match_type = first_row.get("match_type")
            team_home = first_row.get("team_h")
            team_away = first_row.get("team_a")

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
                    "Date": match_date,
                    "Type": match_type,
                    "Home": team_home,
                    "Away": team_away,
                    "Score": full_score_str,
                    "file_id": file_id,
                    "file_name": first_row.get("file_name"),
                    "match_alternative_id": first_row.get("match_alternative_id"),
                }
            )

        matches_df = pd.DataFrame(matches).sort_values(["Date"]).reset_index(drop=True)

        st.dataframe(
            matches_df[["Date", "Type", "Home", "Away", "Score"]],
            width="stretch",
        )

        st.divider()

        st.markdown(f"### Fit Model (alpha={DEFAULT_ALPHA})")
        if last_match_date:
            st.caption(f"Last match date in DB: {last_match_date}")

        if "fitted_params_df" in st.session_state:
            st.info("🔧 Currently using **refitted** parameters (from loaded data)")
        else:
            st.info("📁 Currently using **default** parameters (from CSV file)")

        if st.button(f"Fit model (alpha={DEFAULT_ALPHA})"):
            with st.spinner("Refitting model... this may take a moment..."):
                try:
                    params_df = refit_model_on_current_data(df, alpha=DEFAULT_ALPHA)
                    st.session_state["fitted_params_df"] = params_df

                    num_teams = len(params_df[params_df["par_type"] == "team"]["team_id"].unique())
                    num_params = len(params_df)
                    st.success("✅ Model refitted successfully!")
                    st.info(f"Fitted {num_params} parameters for {num_teams} teams using {len(df)} rallies.")
                    st.info("💡 The rotation simulator now uses the refitted parameters. Navigate to it to see the updated values.")
                except Exception as e:
                    st.error(f"Error refitting model: {e}")
                    import traceback

                    st.code(traceback.format_exc())

        st.divider()

        st.markdown("### 3. Download Data (from Database)")

        if st.button("Download ALL Matches (Merged CSV)"):
            with st.spinner("Fetching ALL matches from Database..."):
                try:
                    full_df = load_full_data_from_db()

                    if not full_df.empty:
                        csv_all = full_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Click to Download MERGED CSV",
                            data=csv_all,
                            file_name="all_matches_full.csv",
                            mime="text/csv",
                            key="dl_all",
                        )
                    else:
                        st.warning("No data found in database.")

                except Exception as e:
                    st.error(f"Error fetching data: {e}")

        st.markdown("#### Download Single Match")

        match_options = {}
        for _, row in matches_df.iterrows():
            label = f"{row['Date']} | {row['Home']} vs {row['Away']}"
            match_id = row.get("match_alternative_id")
            if match_id:
                match_options[label] = match_id

        if match_options:
            selected_label = st.selectbox("Select Match", options=list(match_options.keys()))
            if selected_label:
                selected_match_id = match_options[selected_label]

                if st.button("Prepare CSV for Selected Match"):
                    with st.spinner("Fetching match data..."):
                        try:
                            match_df = load_match_data_from_db(selected_match_id)

                            if not match_df.empty:
                                csv = match_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="Download CSV",
                                    data=csv,
                                    file_name=f"match_{selected_match_id}.csv",
                                    mime="text/csv",
                                    key=f"dl_single_{selected_match_id}",
                                )
                            else:
                                st.warning(f"No data found for match ID: {selected_match_id}")

                        except Exception as e:
                            st.error(f"Error processing match: {e}")
