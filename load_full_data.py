from __future__ import annotations

import os
import json
import pandas as pd
from datetime import datetime

import datavolley as dv

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_files_sorted(dir_path):
    """Return a sorted list of filenames in a directory."""
    return sorted(
        [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    )

import re
from datetime import datetime, date, time
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np


import pandas as pd
import numpy as np

def add_rally_metadata(
    df,
    set_col="set_number",
    time_col="video_time",
    skill_col="skill",
    team_col="team",
    home_team_col="home_team",
    away_team_col="visiting_team",
    home_score_col="home_team_score",
    away_score_col="visiting_team_score",
    code_col="code",
    sort_by_video_time=False
):
    df = df.copy()
    df["_orig_idx"] = np.arange(len(df))

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[set_col] = pd.to_numeric(df[set_col], errors="coerce").astype("Int64")
    for c in (home_score_col, away_score_col):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if sort_by_video_time:
        df[time_col] = df[time_col].fillna(method='ffill').fillna(0)
        df = df.sort_values([set_col, time_col, "_orig_idx"], kind="mergesort")

    n = len(df)
    rally_numbers = np.zeros(n, dtype=int)
    point_won_by_col = [None] * n

    serve_records = []   # (set, rally, serving_team)
    point_records = []   # (set, rally, point_won_by)

    current_set = None
    current_rally = 0
    last_home_score = None
    last_away_score = None

    for i, row in df.iterrows():
        row_set = row[set_col]

        if current_set is None or row_set != current_set:
            current_set = row_set
            current_rally = 1
            last_home_score = row[home_score_col]
            last_away_score = row[away_score_col]

        rally_numbers[i] = current_rally

        if row[skill_col] == "Serve":
            serve_team = row[team_col] if pd.notna(row[team_col]) and row[team_col] != "" else None
            if serve_team:
                serve_records.append((row_set, current_rally, serve_team))

        if row[skill_col] == "Point":
            winner = None
            if row[home_score_col] > (last_home_score if last_home_score is not None else -1):
                winner = row[home_team_col]
            elif row[away_score_col] > (last_away_score if last_away_score is not None else -1):
                winner = row[away_team_col]
            else:
                code_val = str(row.get(code_col, ""))
                if code_val.startswith("*p"):
                    winner = row[home_team_col]
                elif code_val.startswith("ap"):
                    winner = row[away_team_col]

            point_won_by_col[i] = winner
            point_records.append((row_set, current_rally, winner))

            current_rally += 1

        last_home_score = row[home_score_col]
        last_away_score = row[away_score_col]

    df["rally_number"] = rally_numbers

    # serving team per rally
    if serve_records:
        serve_df = (
            pd.DataFrame(serve_records, columns=[set_col, "rally_number", "serving_team"])
            .dropna(subset=["serving_team"])
            .drop_duplicates([set_col, "rally_number"])
        )
        df = df.merge(serve_df, on=[set_col, "rally_number"], how="left")
    else:
        df["serving_team"] = np.nan

    df["serving_team"] = df.groupby([set_col, "rally_number"])["serving_team"].ffill().bfill()

    # point winner per rally
    if point_records:
        point_df = pd.DataFrame(point_records, columns=[set_col, "rally_number", "point_won_by"])
        point_df = point_df.dropna(subset=["point_won_by"])
        if not point_df.empty:
            point_df = point_df.drop_duplicates([set_col, "rally_number"])
            df = df.merge(point_df, on=[set_col, "rally_number"], how="left", suffixes=("", "_from_point"))
            if "point_won_by_from_point" in df.columns:
                df["point_won_by"] = df["point_won_by_from_point"].combine_first(df.get("point_won_by"))
                df.drop(columns=["point_won_by_from_point"], inplace=True)
        else:
            df["point_won_by"] = np.nan
    else:
        df["point_won_by"] = np.nan

    df["point_won_by"] = df.groupby([set_col, "rally_number"])["point_won_by"].ffill().bfill()

    # receiving team
    df["receiving_team"] = np.where(
        df["serving_team"] == df[home_team_col],
        df[away_team_col],
        np.where(
            df["serving_team"] == df[away_team_col],
            df[home_team_col],
            np.nan,
        ),
    )

    # ===== possessions (updated) =====
    df["possession_number"] = 0
    for (s, r), g in df.groupby([set_col, "rally_number"], sort=False):
        # find first Serve in this rally
        serve_mask = g[skill_col].eq("Serve")
        first_serve_idx = serve_mask.idxmax() if serve_mask.any() else None

        current_pos = 0
        current_team = None
        poss_vals = []

        for idx, row in g.iterrows():
            # before the serve -> stay 0
            if first_serve_idx is not None and idx < first_serve_idx:
                poss_vals.append(0)
                continue

            # after (or at) the serve we start counting
            row_team = row[team_col]

            # if the serve row has no team, fall back to serving_team
            if first_serve_idx is not None and idx == first_serve_idx and (pd.isna(row_team) or row_team == ""):
                row_team = row["serving_team"]

            # empty team even after serve -> keep current possession
            if pd.isna(row_team) or row_team == "":
                poss_vals.append(current_pos)
                continue

            if current_pos == 0:
                current_pos = 1
                current_team = row_team
            else:
                if row_team != current_team:
                    current_pos += 1
                    current_team = row_team
            poss_vals.append(current_pos)

        df.loc[g.index, "possession_number"] = poss_vals
    # ===== end possessions =====

    df = df.sort_values("_orig_idx").drop(columns=["_orig_idx"])
    return df


import re
from datetime import datetime
from typing import Optional, Tuple


def _try_parse_date(date_string: str) -> str:
    """
    Attempt to parse a date string from common formats.
    Returns the date in YYYY-MM-DD format if successful, otherwise the original string.
    """
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_string, fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return date_string


def extract_match_date_and_type(path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a Data Volley .dvw-like file, return (match_date_str, match_type_str).

    match_date_str is returned in "YYYY-MM-DD" format if parsable, otherwise as the original string.
    match_type_str is e.g. "Amichevole".

    If something is missing, returns None for that part.
    """
    match_date = None
    match_type = None

    with open(path, "r", encoding="cp1252", errors="ignore") as f:
        lines = [line.strip() for line in f]

    # 1) try to get the line right after [3MATCH]
    match_section_idx = None
    for i, line in enumerate(lines):
        if line.strip().upper() == "[3DATAVOLLEYSCOUT]":
            continue
        if line.strip().upper() == "[3MATCH]":
            match_section_idx = i
            break

    if match_section_idx is not None:
        for j in range(match_section_idx + 1, len(lines)):
            line = lines[j].strip()
            if not line:
                continue
            parts = line.split(";")
            if parts:
                if len(parts) >= 1 and parts[0].strip():
                    raw_date = parts[0].strip()
                    match_date = _try_parse_date(raw_date)

                if len(parts) >= 5 and parts[4].strip():
                    mt = parts[4].strip()
                    mt = re.sub(r"[\x00-\x1f]", "", mt)
                    match_type = mt
                break

    # 2) fallback for date: use GENERATOR-DAY if we didn't find the match date
    if match_date is None:
        for line in lines:
            if line.startswith("GENERATOR-DAY:"):
                raw = line.split("GENERATOR-DAY:", 1)[1].strip()
                raw_date = raw.split()[0]
                match_date = _try_parse_date(raw_date)
                break


    return match_date, match_type


ORDERED_COLS = [
    # 1) Match identification
    "match_id",
    "match_alternative_id",
    "match_type",
    "match_date",

    # 2) Teams
    "home_team_id",
    "home_team",
    "visiting_team_id",
    "visiting_team",

    # 3) Set and score
    "set_number",
    "home_team_score",
    "visiting_team_score",

    # 4) Rally metadata
    "rally_number",
    "point_won_by",
    "serving_team",
    "receiving_team",
    "setter_position",
    "home_setter_position",
    "visiting_setter_position",
    "possession_number",

    # 5) Event / DV code
    "video_time",
    "code",
    "custom_code",
    "point_phase",
    "attack_phase",

    # 6) Actor
    "team",
    "player_id",
    "player_name",
    "player_number",

    # 7) Skill/action details
    "skill",
    "skill_type",
    "skill_subtype",
    "evaluation_code",

    # 8) Technical context
    "attack_code",
    "set_code",
    "set_type",

    # 9) Zones / counts
    "start_zone",
    "end_zone",
    "end_subzone",
    "num_players_numeric",

    # 10) Coordinates
    "start_coordinate",
    "mid_coordinate",
    "end_coordinate",
    "start_coordinate_x",
    "start_coordinate_y",
    "mid_coordinate_x",
    "mid_coordinate_y",
    "end_coordinate_x",
    "end_coordinate_y",

    # 11) Lineups
    "home_p1", "home_p2", "home_p3", "home_p4", "home_p5", "home_p6",
    "visiting_p1", "visiting_p2", "visiting_p3", "visiting_p4", "visiting_p5", "visiting_p6",
]


def process_dv_file(path: str) -> pd.DataFrame:
    """Read one DV file, enrich, and return a clean DataFrame."""
    raw = dv.read_dv(path)
    df = pd.DataFrame(raw)

    # drop old rally columns if present
    cols_to_exclude = [
        "rally_number", "point_won_by", "serving_team",
        "receiving_team", "possession_number"
    ]
    df = df[[c for c in df.columns if c not in cols_to_exclude]]

    # add rally metadata
    df = add_rally_metadata(df)

    # match-level info
    match_date_str, match_type = extract_match_date_and_type(path)

    df["match_date"] = match_date_str
    df["match_type"] = match_type

    # build alternative id (match identifier based on date and team IDs)
    df["match_alternative_id"] = (
        df["match_date"].astype(str)
        + " | "
        + df["home_team_id"].astype(str)
        + " | "
        + df["visiting_team_id"].astype(str)
    )

    # order columns (put known ones first)
    existing = [c for c in ORDERED_COLS if c in df.columns]
    extra = [c for c in df.columns if c not in existing]
    df = df[existing + extra]

    return df


def concat_align_and_save(dfs: list[pd.DataFrame], output_path: str) -> pd.DataFrame:
    """
    Take a list of DataFrames, align them to the union of columns,
    concatenate, save to CSV, and return the final DataFrame.
    """
    # union of columns
    all_cols = set()
    for d in dfs:
        all_cols.update(d.columns)
    all_cols = list(all_cols)

    # align
    aligned = [d.reindex(columns=all_cols) for d in dfs]

    # concat
    final_df = pd.concat(aligned, ignore_index=True)

    # ========================================================================
    # MANUAL FIX / AD-HOC DATA CLEANING
    # ========================================================================
    # Issue: "Conad Reggio Emilia" appears with two different team_ids in the data.
    # This causes duplicate entries in team summary and breaks analysis.
    # Fix: Map all instances to a single canonical team_id.
    # TODO: Investigate root cause in data source and fix upstream if possible.
    # ========================================================================
    
    # Identify the team name(s) affected
    TEAM_NAME_TO_FIX = "Conad Reggio Emilia"
    
    # Find all team_ids associated with this team name
    affected_home = final_df[final_df["home_team"] == TEAM_NAME_TO_FIX]["home_team_id"].unique()
    affected_away = final_df[final_df["visiting_team"] == TEAM_NAME_TO_FIX]["visiting_team_id"].unique()
    all_affected_ids = list(set(list(affected_home) + list(affected_away)))
    
    if len(all_affected_ids) > 1:
        # Use the first ID as canonical (or specify explicitly)
        canonical_id = str(min(all_affected_ids))  # Use minimum for consistency
        
        logger.warning(f"⚠️  MANUAL FIX: Team '{TEAM_NAME_TO_FIX}' has multiple IDs: {all_affected_ids}")
        logger.warning(f"⚠️  Consolidating all to canonical ID: {canonical_id}")
        
        # Replace all occurrences in both home and away columns
        for old_id in all_affected_ids:
            if old_id != canonical_id:
                final_df.loc[final_df["home_team_id"] == old_id, "home_team_id"] = canonical_id
                final_df.loc[final_df["visiting_team_id"] == old_id, "visiting_team_id"] = canonical_id
                logger.info(f"   Replaced team_id {old_id} → {canonical_id}")
        
        # Rebuild match_alternative_id after team ID correction
        if "match_alternative_id" in final_df.columns:
            final_df["match_alternative_id"] = (
                final_df["match_date"].astype(str)
                + " | "
                + final_df["home_team_id"].astype(str)
                + " | "
                + final_df["visiting_team_id"].astype(str)
            )
    
    # Sort by match_date if available
    if "match_date" in final_df.columns:
        final_df = final_df.sort_values(by=["match_date"]).reset_index(drop=True)

    # save
    final_df.to_csv(output_path, index=False)

    return final_df


import tempfile

def sanitize_dv_content(content: str) -> str:
    """
    Sanitize DV content to prevent parser crashes.
    Specifically, comment out scout lines where the code is too short (< 6 chars),
    which causes IndexError in datavolley.
    """
    lines = content.splitlines()
    sanitized_lines = []
    in_scout = False
    
    for line in lines:
        stripped = line.strip()
        if stripped == "[3SCOUT]":
            in_scout = True
            sanitized_lines.append(line)
            continue
            
        if in_scout:
            # Check if we hit the next section (sections start with [)
            if stripped.startswith("["):
                in_scout = False
                sanitized_lines.append(line)
                continue
                
            # Process scout line
            # Format: code;...
            parts = line.split(";")
            if parts and parts[0]:
                code = parts[0].strip()
                # Check for short codes that are not comments
                # The parser bug is specifically: if len(code) > 4: access code[5].
                # This crashes ONLY for len(code) == 5.
                # Codes with len <= 4 are safe (the if is skipped).
                # Codes with len >= 6 are safe (code[5] exists).
                if code and not code.startswith("*") and len(code) == 5:
                    # Log if possible, but we are in a helper. 
                    # Just comment it out to be safe.
                    # logging.warning(f"Sanitizing malformed code: {code}")
                    sanitized_lines.append("*" + line)
                else:
                    sanitized_lines.append(line)
            else:
                sanitized_lines.append(line)
        else:
            sanitized_lines.append(line)
            
    return "\n".join(sanitized_lines)

def process_dv_file_content(file_content: str | bytes, file_name: str = "temp.dvw") -> pd.DataFrame:
    """
    Process DV file content by writing it to a temporary file (preserving filename) 
    and using the existing processing logic.
    """
    # Decode bytes if needed for sanitization
    if isinstance(file_content, bytes):
        try:
            content_str = file_content.decode("cp1252", errors="ignore")
        except:
            content_str = file_content.decode("utf-8", errors="ignore")
    else:
        content_str = file_content

    # Sanitize content
    content_str = sanitize_dv_content(content_str)

    # Create a temporary directory so we can use the real filename
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, file_name)
        
        # Write content (always as cp1252 for datavolley)
        with open(temp_path, "w", encoding="cp1252", errors="ignore") as f:
            f.write(content_str)

        # Process
        df = process_dv_file(temp_path)
        return df

import streamlit as st
from gdrive_utils import list_files_in_folder, read_file_content
from db_utils import get_existing_match_ids, get_existing_file_ids_from_touches, upload_touches, fetch_all_touches, fetch_touches_by_match_id

def update_database_full(folder_ids: list[str], progress_callback=None) -> list[str]:
    """
    Checks for new files in Google Drive that are not in the Supabase DB (touch_level_data),
    processes them, and uploads the data.
    Returns a list of names of the matches/files that were uploaded.
    """
    logging.info("Starting database update for FULL data...")
    uploaded_matches = []
    
    # 1. Get existing file IDs from DB
    existing_file_ids = get_existing_file_ids_from_touches()
    logging.info(f"Found {len(existing_file_ids)} files already in database.")
    
    # 2. List files in GDrive
    all_dvw_files = []
    for folder_id in folder_ids:
        try:
            files = list_files_in_folder(folder_id)
            dvw_files = [f for f in files if f['name'].lower().endswith('.dvw')]
            all_dvw_files.extend(dvw_files)
        except Exception as e:
            logging.error(f"Error scanning folder {folder_id}: {e}")
            
    # 3. Filter for missing files
    missing_files = [f for f in all_dvw_files if f['id'] not in existing_file_ids]
    logging.info(f"Found {len(missing_files)} new files to process.")
    
    if not missing_files:
        logging.info("Database is up to date.")
        return uploaded_matches
    
    # 4. Process and upload missing files
    total_files = len(missing_files)
    for i, f in enumerate(missing_files):
        msg = f"Processing {f['name']} ({i+1}/{total_files})"
        logging.info(msg)
        if progress_callback:
            progress_callback(i+1, total_files, msg)
        
        try:
            content = read_file_content(f['id'])
            # Process content
            df_temp = process_dv_file_content(content, f['name'])
            
            if df_temp.empty:
                logging.warning(f"No data found in {f['name']}")
                continue
                
            # Add file metadata
            df_temp['file_id'] = f['id']
            df_temp['file_name'] = f['name']
            
            # --- Apply Manual Fixes (Reggio Emilia) ---
            # Copied from concat_align_and_save logic
            TEAM_NAME_TO_FIX = "Conad Reggio Emilia"
            if "home_team" in df_temp.columns and "visiting_team" in df_temp.columns:
                affected_home = df_temp[df_temp["home_team"] == TEAM_NAME_TO_FIX]["home_team_id"].unique()
                affected_away = df_temp[df_temp["visiting_team"] == TEAM_NAME_TO_FIX]["visiting_team_id"].unique()
                all_affected_ids = list(set(list(affected_home) + list(affected_away)))
                
                if len(all_affected_ids) > 1:
                    canonical_id = str(min(all_affected_ids))
                    logging.warning(f"Applying fix for {TEAM_NAME_TO_FIX}: {all_affected_ids} -> {canonical_id}")
                    for old_id in all_affected_ids:
                        if old_id != canonical_id:
                            df_temp.loc[df_temp["home_team_id"] == old_id, "home_team_id"] = canonical_id
                            df_temp.loc[df_temp["visiting_team_id"] == old_id, "visiting_team_id"] = canonical_id
                    
                    # Rebuild match_alternative_id
                    df_temp["match_alternative_id"] = (
                        df_temp["match_date"].astype(str)
                        + " | "
                        + df_temp["home_team_id"].astype(str)
                        + " | "
                        + df_temp["visiting_team_id"].astype(str)
                    )

            # --- Validate and Fill Missing Values for NOT NULL columns ---
            # match_id
            if "match_id" not in df_temp.columns or df_temp["match_id"].isnull().all():
                # Fallback to match_alternative_id if match_id is missing
                df_temp["match_id"] = df_temp["match_alternative_id"]
            else:
                df_temp["match_id"] = df_temp["match_id"].fillna(df_temp["match_alternative_id"])
            
            # match_date
            if "match_date" not in df_temp.columns:
                 # Should have been set by process_dv_file, but just in case
                 df_temp["match_date"] = "1970-01-01"
            df_temp["match_date"] = df_temp["match_date"].fillna("1970-01-01")

            # set_number
            if "set_number" not in df_temp.columns:
                df_temp["set_number"] = 1
            df_temp["set_number"] = df_temp["set_number"].fillna(1).astype(int)

            # scores
            for col in ["home_team_score", "visiting_team_score"]:
                if col not in df_temp.columns:
                    df_temp[col] = 0
                df_temp[col] = df_temp[col].fillna(0).astype(int)
            
            # rally_number
            if "rally_number" not in df_temp.columns:
                df_temp["rally_number"] = 0
            df_temp["rally_number"] = df_temp["rally_number"].fillna(0).astype(int)

            # possession_number
            if "possession_number" not in df_temp.columns:
                df_temp["possession_number"] = 0
            df_temp["possession_number"] = df_temp["possession_number"].fillna(0).astype(int)

            # team
            if "team" not in df_temp.columns:
                df_temp["team"] = "Unknown"
            df_temp["team"] = df_temp["team"].fillna("Unknown")

            # Ensure match_alternative_id is not null
            df_temp["match_alternative_id"] = df_temp["match_alternative_id"].fillna("Unknown")

            # --- Explicitly cast Integer columns to Int64 (nullable int) ---
            # This ensures 12.0 becomes 12, and NaN becomes <NA> (which we handle in db_utils)
            int_cols = [
                "set_number", "home_team_score", "visiting_team_score", "rally_number", 
                "possession_number", "setter_position", "home_setter_position", 
                "visiting_setter_position", "player_number", "num_players_numeric",
                "home_p1", "home_p2", "home_p3", "home_p4", "home_p5", "home_p6",
                "visiting_p1", "visiting_p2", "visiting_p3", "visiting_p4", "visiting_p5", "visiting_p6"
            ]
            
            for col in int_cols:
                if col in df_temp.columns:
                    # Coerce to numeric first (handles strings like "12"), then cast to Int64
                    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').astype('Int64')

            # --- Add unique_row_id (0..N) as requested for Primary Key ---
            df_temp.insert(0, "unique_row_id", range(len(df_temp)))
            df_temp["unique_row_id"] = df_temp["unique_row_id"].astype('Int64')

            # Upload to DB
            upload_touches(df_temp)
            logging.info(f"Uploaded {len(df_temp)} touches for {f['name']}")
            uploaded_matches.append(f['name'])
            
            # Add to local cache of existing IDs
            existing_file_ids.add(f['id'])
            
        except Exception as e:
            logging.error(f"Error processing/uploading file {f['name']}: {e}")
            
    return uploaded_matches

SQL_ORDERED_COLS = [
    "unique_row_id",
    "match_id", "match_alternative_id", "match_type", "match_date",
    "home_team_id", "home_team", "visiting_team_id", "visiting_team",
    "set_number", "home_team_score", "visiting_team_score", "rally_number",
    "point_won_by", "serving_team", "receiving_team",
    "setter_position", "home_setter_position", "visiting_setter_position",
    "possession_number", "video_time",
    "code", "custom_code", "point_phase", "attack_phase",
    "team", "player_id", "player_name", "player_number",
    "skill", "skill_type", "skill_subtype", "evaluation_code", "attack_code", "set_code", "set_type",
    "start_zone", "end_zone", "end_subzone", "num_players_numeric",
    "start_coordinate", "mid_coordinate", "end_coordinate",
    "start_coordinate_x", "start_coordinate_y", "mid_coordinate_x", "mid_coordinate_y", "end_coordinate_x", "end_coordinate_y",
    "home_p1", "home_p2", "home_p3", "home_p4", "home_p5", "home_p6",
    "visiting_p1", "visiting_p2", "visiting_p3", "visiting_p4", "visiting_p5", "visiting_p6",
    "file_id", "file_name",
    "created_by", "create_datetime"
]

def load_full_data_from_db() -> pd.DataFrame:
    """
    Loads the full dataset from Supabase.
    """
    logging.info("Fetching FULL data from Supabase...")
    df = fetch_all_touches()
    
    if not df.empty:
        # Apply global fix if needed (though we fix on upload, legacy data might need it)
        TEAM_NAME_TO_FIX = "Conad Reggio Emilia"
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
        
        # Sort
        # Ensure unique_row_id is int for sorting
        if 'unique_row_id' in df.columns:
            df['unique_row_id'] = pd.to_numeric(df['unique_row_id'], errors='coerce').fillna(0).astype(int)
            
        sort_cols = ['match_date', 'file_id', 'unique_row_id']
        # Check if cols exist
        existing_sort_cols = [c for c in sort_cols if c in df.columns]
        if existing_sort_cols:
            df = df.sort_values(by=existing_sort_cols).reset_index(drop=True)
            
        # Reorder columns to match SQL table
        # Only include columns that exist in the DF (ignore missing ones like created_by if not fetched or not in DF)
        cols_to_use = [c for c in SQL_ORDERED_COLS if c in df.columns]
        # Append any extra columns that might be in DF but not in SQL list (just in case)
        extra_cols = [c for c in df.columns if c not in cols_to_use]
        df = df[cols_to_use + extra_cols]
            
    return df

def load_match_data_from_db(match_id: str) -> pd.DataFrame:
    """
    Loads data for a specific match from Supabase.
    """
    logging.info(f"Fetching data for match {match_id} from Supabase...")
    df = fetch_touches_by_match_id(match_id)
    
    if not df.empty:
        # Apply global fix if needed
        TEAM_NAME_TO_FIX = "Conad Reggio Emilia"
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
        
        # Sort
        if "unique_row_id" in df.columns:
             df = df.sort_values(by=["unique_row_id"]).reset_index(drop=True)
        elif "rally_number" in df.columns:
             df = df.sort_values(by=["rally_number"]).reset_index(drop=True)

        # Reorder columns to match SQL table
        existing_cols = [c for c in SQL_ORDERED_COLS if c in df.columns]
        df = df[existing_cols]
        
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    output_path = "./clean_data/clean_full_data.csv"

    # Check if GDrive is configured in secrets
    folder_ids = []
    if "gdrive" in st.secrets:
        if "folder_ids" in st.secrets["gdrive"]:
            folder_ids = st.secrets["gdrive"]["folder_ids"]
        elif "folder_id" in st.secrets["gdrive"]:
            folder_ids = [st.secrets["gdrive"]["folder_id"]]

    if folder_ids:
        # 1. Update Database
        update_database_full(folder_ids)
        
        # 2. Load from Database
        final_df = load_full_data_from_db()
        
        if not final_df.empty:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logging.info('Saving file : {}'.format(output_path))
            final_df.to_csv(output_path, index=False)
            print(f"Loaded {len(final_df)} rows.")
        else:
            logging.warning("No data found in database.")
    else:
        # Fallback to local if no secrets (or for testing)
        logging.warning("No GDrive secrets found. Using local ./data folder.")
        input_dir_path = "./data"
        if os.path.exists(input_dir_path):
            files = list_files_sorted(input_dir_path)
            per_file_dfs = []
            for i, fn in enumerate(files):
                logging.info("Processing file %s (%d/%d)", fn, i + 1, len(files))
                try:
                    with open(fn, "r", encoding="cp1252", errors="ignore") as f:
                        content = f.read()
                    file_name = os.path.basename(fn)
                    df_temp = process_dv_file_content(content, file_name)
                    per_file_dfs.append(df_temp)
                except Exception as e:
                    logging.error(f"Error processing file {fn}: {e}")
                    continue
            if per_file_dfs:
                final_df = concat_align_and_save(per_file_dfs, output_path)
                print(final_df)
