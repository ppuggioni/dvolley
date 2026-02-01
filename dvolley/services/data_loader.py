from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from dvolley.data.dvw_parser import dvw_rallies_to_df
from dvolley.data.full_parser import process_dv_file_content
from dvolley.data.normalization import normalize_date_str
from dvolley.services import db
from dvolley.config import TEAM_NAME_TO_FIX, SQL_ORDERED_COLS
from dvolley.services.gdrive import list_files_in_folder, read_file_content

logger = logging.getLogger(__name__)


def update_database(folder_ids: list[str], progress_callback=None) -> Optional[list[str]]:
    """
    Checks for new files in Google Drive that are not in the Supabase DB,
    processes them, and uploads the data.
    """
    logger.info("Starting database update...")

    existing_file_ids = db.get_existing_file_ids()
    logger.info("Found %s files already in database.", len(existing_file_ids))

    all_dvw_files = []
    for folder_id in folder_ids:
        try:
            files = list_files_in_folder(folder_id)
            dvw_files = [f for f in files if f["name"].lower().endswith(".dvw")]
            all_dvw_files.extend(dvw_files)
        except Exception as e:
            logger.error("Error scanning folder %s: %s", folder_id, e)

    missing_files = [f for f in all_dvw_files if f["id"] not in existing_file_ids]
    logger.info("Found %s new files to process.", len(missing_files))

    if not missing_files:
        logger.info("Database is up to date.")
        return []

    new_files = []
    total_files = len(missing_files)
    for i, f in enumerate(missing_files):
        msg = f"Processing {f['name']} ({i+1}/{total_files})"
        logger.info(msg)
        if progress_callback:
            progress_callback(i + 1, total_files, msg)

        try:
            content = read_file_content(f["id"])
            df_temp = dvw_rallies_to_df(content)

            if df_temp.empty:
                logger.warning("No rallies found in %s", f["name"])
                continue

            df_temp["file_id"] = f["id"]
            df_temp["file_name"] = f["name"]
            if "match_date" in df_temp.columns:
                df_temp["match_date"] = df_temp["match_date"].apply(normalize_date_str)

            df_temp["match_alternative_id"] = (
                df_temp["match_date"].astype(str)
                + " | "
                + df_temp["team_id_h"].astype(str)
                + " | "
                + df_temp["team_id_a"].astype(str)
            )

            db.upload_rallies(df_temp)
            logger.info("Uploaded %s rallies for %s", len(df_temp), f["name"])
            new_files.append(f["name"])

        except Exception as e:
            logger.error("Error processing/uploading file %s: %s", f["name"], e)

    return new_files


def load_data_from_db() -> pd.DataFrame:
    """
    Loads the full rally dataset from Supabase.
    """
    logger.info("Fetching data from Supabase...")
    df = db.fetch_all_rallies()

    if not df.empty:
        if "match_date" in df.columns:
            df["match_date"] = df["match_date"].apply(normalize_date_str)
        if "rally_idx" in df.columns:
            df["rally_idx"] = pd.to_numeric(df["rally_idx"], errors="coerce").fillna(0).astype(int)

        sort_cols = ["match_date", "file_id", "rally_idx"]
        existing_sort_cols = [c for c in sort_cols if c in df.columns]
        if existing_sort_cols:
            df = df.sort_values(by=existing_sort_cols).reset_index(drop=True)

        if "team_h" in df.columns and "team_a" in df.columns:
            affected_home = df[df["team_h"] == TEAM_NAME_TO_FIX]["team_id_h"].unique()
            affected_away = df[df["team_a"] == TEAM_NAME_TO_FIX]["team_id_a"].unique()
            all_affected_ids = list(set(list(affected_home) + list(affected_away)))

            if len(all_affected_ids) > 1:
                canonical_id = str(min(all_affected_ids))
                logger.warning(
                    "Applying fix for %s: %s -> %s",
                    TEAM_NAME_TO_FIX,
                    all_affected_ids,
                    canonical_id,
                )
                for old_id in all_affected_ids:
                    if old_id != canonical_id:
                        df.loc[df["team_id_h"] == old_id, "team_id_h"] = canonical_id
                        df.loc[df["team_id_a"] == old_id, "team_id_a"] = canonical_id

                df["match_alternative_id"] = (
                    df["match_date"].astype(str)
                    + " | "
                    + df["team_id_h"].astype(str)
                    + " | "
                    + df["team_id_a"].astype(str)
                )

    return df


def update_database_full(folder_ids: list[str], progress_callback=None) -> list[str]:
    """
    Checks for new files in Google Drive that are not in the Supabase DB (touch_level_data),
    processes them, and uploads the data.
    Returns a list of names of the matches/files that were uploaded.
    """
    logger.info("Starting database update for FULL data...")
    uploaded_matches = []

    existing_file_ids = db.get_existing_file_ids_from_touches()
    logger.info("Found %s files already in database.", len(existing_file_ids))

    all_dvw_files = []
    for folder_id in folder_ids:
        try:
            files = list_files_in_folder(folder_id)
            dvw_files = [f for f in files if f["name"].lower().endswith(".dvw")]
            all_dvw_files.extend(dvw_files)
        except Exception as e:
            logger.error("Error scanning folder %s: %s", folder_id, e)

    missing_files = [f for f in all_dvw_files if f["id"] not in existing_file_ids]
    logger.info("Found %s new files to process.", len(missing_files))

    if not missing_files:
        logger.info("Database is up to date.")
        return uploaded_matches

    total_files = len(missing_files)
    for i, f in enumerate(missing_files):
        msg = f"Processing {f['name']} ({i+1}/{total_files})"
        logger.info(msg)
        if progress_callback:
            progress_callback(i + 1, total_files, msg)

        try:
            content = read_file_content(f["id"])
            df_temp = process_dv_file_content(content, f["name"])

            if df_temp.empty:
                logger.warning("No data found in %s", f["name"])
                continue

            df_temp["file_id"] = f["id"]
            df_temp["file_name"] = f["name"]
            if "match_date" in df_temp.columns:
                df_temp["match_date"] = df_temp["match_date"].apply(normalize_date_str)

            if "home_team" in df_temp.columns and "visiting_team" in df_temp.columns:
                affected_home = df_temp[df_temp["home_team"] == TEAM_NAME_TO_FIX]["home_team_id"].unique()
                affected_away = df_temp[df_temp["visiting_team"] == TEAM_NAME_TO_FIX]["visiting_team_id"].unique()
                all_affected_ids = list(set(list(affected_home) + list(affected_away)))

                if len(all_affected_ids) > 1:
                    canonical_id = str(min(all_affected_ids))
                    logger.warning("Applying fix for %s: %s -> %s", TEAM_NAME_TO_FIX, all_affected_ids, canonical_id)
                    for old_id in all_affected_ids:
                        if old_id != canonical_id:
                            df_temp.loc[df_temp["home_team_id"] == old_id, "home_team_id"] = canonical_id
                            df_temp.loc[df_temp["visiting_team_id"] == old_id, "visiting_team_id"] = canonical_id

                    df_temp["match_alternative_id"] = (
                        df_temp["match_date"].astype(str)
                        + " | "
                        + df_temp["home_team_id"].astype(str)
                        + " | "
                        + df_temp["visiting_team_id"].astype(str)
                    )

            if "match_id" not in df_temp.columns or df_temp["match_id"].isnull().all():
                df_temp["match_id"] = df_temp["match_alternative_id"]
            else:
                df_temp["match_id"] = df_temp["match_id"].fillna(df_temp["match_alternative_id"])

            if "match_date" not in df_temp.columns:
                df_temp["match_date"] = "1970-01-01"
            df_temp["match_date"] = df_temp["match_date"].fillna("1970-01-01")

            if "set_number" not in df_temp.columns:
                df_temp["set_number"] = 1
            df_temp["set_number"] = df_temp["set_number"].fillna(1).astype(int)

            for col in ["home_team_score", "visiting_team_score"]:
                if col not in df_temp.columns:
                    df_temp[col] = 0
                df_temp[col] = df_temp[col].fillna(0).astype(int)

            if "rally_number" not in df_temp.columns:
                df_temp["rally_number"] = 0
            df_temp["rally_number"] = df_temp["rally_number"].fillna(0).astype(int)

            if "possession_number" not in df_temp.columns:
                df_temp["possession_number"] = 0
            df_temp["possession_number"] = df_temp["possession_number"].fillna(0).astype(int)

            if "team" not in df_temp.columns:
                df_temp["team"] = "Unknown"
            df_temp["team"] = df_temp["team"].fillna("Unknown")

            df_temp["match_alternative_id"] = df_temp["match_alternative_id"].fillna("Unknown")

            int_cols = [
                "set_number",
                "home_team_score",
                "visiting_team_score",
                "rally_number",
                "possession_number",
                "setter_position",
                "home_setter_position",
                "visiting_setter_position",
                "player_number",
                "num_players_numeric",
                "home_p1",
                "home_p2",
                "home_p3",
                "home_p4",
                "home_p5",
                "home_p6",
                "visiting_p1",
                "visiting_p2",
                "visiting_p3",
                "visiting_p4",
                "visiting_p5",
                "visiting_p6",
            ]

            for col in int_cols:
                if col in df_temp.columns:
                    df_temp[col] = pd.to_numeric(df_temp[col], errors="coerce").astype("Int64")

            df_temp.insert(0, "unique_row_id", range(len(df_temp)))
            df_temp["unique_row_id"] = df_temp["unique_row_id"].astype("Int64")

            db.upload_touches(df_temp)
            logger.info("Uploaded %s touches for %s", len(df_temp), f["name"])
            uploaded_matches.append(f["name"])

            existing_file_ids.add(f["id"])

        except Exception as e:
            logger.error("Error processing/uploading file %s: %s", f["name"], e)

    return uploaded_matches


def load_full_data_from_db() -> pd.DataFrame:
    """
    Loads the full dataset from Supabase.
    """
    logger.info("Fetching FULL data from Supabase...")
    df = db.fetch_all_touches()

    if not df.empty:
        if "home_team" in df.columns and "visiting_team" in df.columns:
            affected_home = df[df["home_team"] == TEAM_NAME_TO_FIX]["home_team_id"].unique()
            affected_away = df[df["visiting_team"] == TEAM_NAME_TO_FIX]["visiting_team_id"].unique()
            all_affected_ids = list(set(list(affected_home) + list(affected_away)))

            if len(all_affected_ids) > 1:
                canonical_id = str(min(all_affected_ids))
                for old_id in all_affected_ids:
                    if old_id != canonical_id:
                        df.loc[df["home_team_id"] == old_id, "home_team_id"] = canonical_id
                        df.loc[df["visiting_team_id"] == old_id, "visiting_team_id"] = canonical_id

                df["match_alternative_id"] = (
                    df["match_date"].astype(str)
                    + " | "
                    + df["home_team_id"].astype(str)
                    + " | "
                    + df["visiting_team_id"].astype(str)
                )

        if "match_date" in df.columns:
            df["match_date"] = df["match_date"].apply(normalize_date_str)

        if "unique_row_id" in df.columns:
            df["unique_row_id"] = pd.to_numeric(df["unique_row_id"], errors="coerce").fillna(0).astype(int)

        sort_cols = ["match_date", "file_id", "unique_row_id"]
        existing_sort_cols = [c for c in sort_cols if c in df.columns]
        if existing_sort_cols:
            df = df.sort_values(by=existing_sort_cols).reset_index(drop=True)

        cols_to_use = [c for c in SQL_ORDERED_COLS if c in df.columns]
        extra_cols = [c for c in df.columns if c not in cols_to_use]
        df = df[cols_to_use + extra_cols]

    return df


def load_match_data_from_db(match_id: str) -> pd.DataFrame:
    """
    Loads data for a specific match from Supabase.
    """
    logger.info("Fetching data for match %s from Supabase...", match_id)
    df = db.fetch_touches_by_match_id(match_id)

    if not df.empty:
        if "home_team" in df.columns and "visiting_team" in df.columns:
            affected_home = df[df["home_team"] == TEAM_NAME_TO_FIX]["home_team_id"].unique()
            affected_away = df[df["visiting_team"] == TEAM_NAME_TO_FIX]["visiting_team_id"].unique()
            all_affected_ids = list(set(list(affected_home) + list(affected_away)))

            if len(all_affected_ids) > 1:
                canonical_id = str(min(all_affected_ids))
                for old_id in all_affected_ids:
                    if old_id != canonical_id:
                        df.loc[df["home_team_id"] == old_id, "home_team_id"] = canonical_id
                        df.loc[df["visiting_team_id"] == old_id, "visiting_team_id"] = canonical_id

        if "match_date" in df.columns:
            df["match_date"] = df["match_date"].apply(normalize_date_str)

        if "unique_row_id" in df.columns:
            df = df.sort_values(by=["unique_row_id"]).reset_index(drop=True)
        elif "rally_number" in df.columns:
            df = df.sort_values(by=["rally_number"]).reset_index(drop=True)

        existing_cols = [c for c in SQL_ORDERED_COLS if c in df.columns]
        df = df[existing_cols]

    return df
