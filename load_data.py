import os
import json
import pandas as pd
from datetime import datetime

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_files_sorted(dir_path):
    """Return a sorted list of filenames in a directory."""
    return sorted(
        [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    )

import pandas as pd
import re

import re
import pandas as pd

import re
import pandas as pd


import streamlit as st
from gdrive_utils import list_files_in_folder, read_file_content

import streamlit as st
from gdrive_utils import list_files_in_folder, read_file_content
from db_utils import get_existing_file_ids, upload_rallies, fetch_all_rallies

def dvw_rallies_to_df(file_content: str) -> pd.DataFrame:
    """
    Read a Data Volley DVW-like text file content (already decoded) and return 1 row per rally.
    """
    # -------------------------------------------------------------------------
    # 1) read file content
    # -------------------------------------------------------------------------
    lines = file_content.splitlines()

    match_date = None
    match_type = "Unknown" # Default value
    team_id_h = None
    team_h = None
    team_id_a = None
    team_a = None

    # -------------------------------------------------------------------------
    # 2) parse header blocks robustly (your style: tag on a line, data after)
    # -------------------------------------------------------------------------
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()

        # ----- MATCH -----
        if line == "[3MATCH]":
            # take the next non-empty, non-tag line as the main match line
            j = i + 1
            while j < n and (not lines[j].strip() or lines[j].strip().startswith("[")):
                # rare case: empty line right after, skip
                j += 1
            if j < n:
                match_line = lines[j].strip()
                parts = [p.strip() for p in match_line.split(";")]
                # example:
                # 0 date  -> 08/10/2025
                # 1 time  -> 20.30.00
                # 2 season
                # 3 competition -> Regular Season ...
                # 4 match type  -> Amichevole
                if len(parts) > 0:
                    raw_date = parts[0]  # "08/10/2025"
                    try:
                        match_date = datetime.strptime(raw_date, "%d/%m/%Y").strftime("%Y-%m-%d")
                    except ValueError:
                        match_date = raw_date
                if len(parts) > 4:
                    match_type = parts[4]  # "Amichevole"
            i = j  # continue from here
        # ----- TEAMS -----
        elif line == "[3TEAMS]":
            # next line = home team
            # next line after that = away team
            if i + 1 < n:
                home_line = lines[i + 1].strip()
                home_parts = [p.strip() for p in home_line.split(";")]
                if len(home_parts) >= 2:
                    team_id_h = str(home_parts[0])  # Always string
                    team_h = home_parts[1]
            if i + 2 < n:
                away_line = lines[i + 2].strip()
                # make sure it's not a new tag
                if not away_line.startswith("["):
                    away_parts = [p.strip() for p in away_line.split(";")]
                    if len(away_parts) >= 2:
                        team_id_a = str(away_parts[0])  # Always string
                        team_a = away_parts[1]
            # skip ahead
            i += 2
        i += 1

    # -------------------------------------------------------------------------
    # 3) walk through scout events
    # -------------------------------------------------------------------------
    rows = []

    inside_scout = False

    current_set = 1
    sets_h = 0
    sets_a = 0
    pts_h = 0
    pts_a = 0

    # current setter positions BEFORE the next rally
    home_setter_pos = 0 # Default to 0
    away_setter_pos = 0 # Default to 0

    # serving team for the upcoming rally
    current_server_team = None  # 'h' or 'a'

    last_rally_idx = None  # to patch post_set_won_* at end of set

    def process_event(ev: str):
        nonlocal current_set, sets_h, sets_a, pts_h, pts_a
        nonlocal home_setter_pos, away_setter_pos
        nonlocal current_server_team, last_rally_idx

        ev = ev.strip()
        if not ev:
            return

        # -------------------------------------------------------------
        # set end marker
        # -------------------------------------------------------------
        m_endset = re.match(r"^\*\*(\d+)set", ev, flags=re.IGNORECASE)
        if m_endset:
            # finalize this set on the last rally
            if last_rally_idx is not None:
                if pts_h > pts_a:
                    sets_h += 1
                elif pts_a > pts_h:
                    sets_a += 1
                rows[last_rally_idx]["post_set_won_h"] = sets_h
                rows[last_rally_idx]["post_set_won_a"] = sets_a

            # move to next set
            current_set = int(m_endset.group(1)) + 1
            pts_h = 0
            pts_a = 0
            home_setter_pos = 0
            away_setter_pos = 0
            current_server_team = None
            last_rally_idx = None
            return

        # -------------------------------------------------------------
        # setter positions (home)
        #  *z6...
        #  *z6>LUp...
        # -------------------------------------------------------------
        m_home_z = re.match(r"^\*z([1-6])", ev)
        if m_home_z:
            home_setter_pos = int(m_home_z.group(1))
            return

        # -------------------------------------------------------------
        # setter positions (away)
        #  az5...
        #  az5>LUp...
        # -------------------------------------------------------------
        m_away_z = re.match(r"^az([1-6])", ev)
        if m_away_z:
            away_setter_pos = int(m_away_z.group(1))
            return

        # -------------------------------------------------------------
        # serving detection (must come before the rally point line)
        #   *06S..., *10SQ..., a08SM..., a02SQ..., etc.
        # -------------------------------------------------------------
        if re.match(r"^\*\d+S", ev) or re.match(r"^\*\d+SQ", ev):
            current_server_team = "h"
            return
        if re.match(r"^a\d+S", ev) or re.match(r"^a\d+SQ", ev):
            current_server_team = "a"
            return

        # -------------------------------------------------------------
        # scoreboard / rally lines
        # -------------------------------------------------------------
        m_home_p = re.match(r"^\*p(\d+):(\d+)", ev)
        m_away_p = re.match(r"^ap(\d+):(\d+)", ev)

        if not m_home_p and not m_away_p:
            # not a rally, ignore
            return

        if m_home_p:
            new_h = int(m_home_p.group(1))
            new_a = int(m_home_p.group(2))
        else:
            new_h = int(m_away_p.group(1))
            new_a = int(m_away_p.group(2))

        # pre
        pre_point_h = pts_h
        pre_point_a = pts_a
        pre_set_h = sets_h
        pre_set_a = sets_a

        # who won
        if new_h > pts_h:
            winner = "h"
        elif new_a > pts_a:
            winner = "a"
        else:
            winner = "h" if ev.startswith("*p") else "a"

        # serving team for this rally
        if current_server_team is None:
            serving_team = winner  # fallback
        else:
            serving_team = current_server_team

        serve_h = 1 if serving_team == "h" else 0
        serve_a = 1 if serving_team == "a" else 0

        row = {
            "match_type": match_type,
            "match_date": match_date,
            "team_id_h": team_id_h,
            "team_id_a": team_id_a,
            "team_h": team_h,
            "team_a": team_a,
            "set_number": current_set,
            "pre_set_won_h": pre_set_h,
            "pre_set_won_a": pre_set_a,
            "pre_point_won_h": pre_point_h,
            "pre_point_won_a": pre_point_a,
            "p_h": home_setter_pos,
            "p_a": away_setter_pos,
            "post_set_won_h": sets_h,
            "post_set_won_a": sets_a,
            "post_point_won_h": new_h,
            "post_point_won_a": new_a,
            "point_won_h": 1 if winner == "h" else 0,
            "point_won_a": 1 if winner == "a" else 0,
            "point_won_team": winner,
            "serve_h": serve_h,
            "serve_a": serve_a,
            "serve_team": serving_team,
            "rally_idx": len(rows),
        }
        rows.append(row)
        last_rally_idx = len(rows) - 1

        # update running score
        pts_h = new_h
        pts_a = new_a

        # after a point, DV will tell us new z-lines, so we forget serve
        current_server_team = None

    # -------------------------------------------------------------------------
    # 4) run through file and feed events
    # -------------------------------------------------------------------------
    for line in lines:
        if "[3SCOUT]" in line:
            inside_scout = True
            # may have events on same line
            after = line.split("[3SCOUT]", 1)[1].strip()
            if after:
                for ev in after.split():
                    process_event(ev)
            continue

        if not inside_scout:
            continue

        # stop at next section
        if line.strip().startswith("[") and not line.strip().startswith("[3SCOUT]"):
            break

        for ev in line.strip().split():
            process_event(ev)

    # -------------------------------------------------------------------------
    # 5) make dataframe
    # -------------------------------------------------------------------------
    return pd.DataFrame(rows)


def update_database(folder_ids: list[str], progress_callback=None):
    """
    Checks for new files in Google Drive that are not in the Supabase DB,
    processes them, and uploads the data.
    """
    logging.info("Starting database update...")
    
    # 1. Get existing file IDs from DB
    existing_file_ids = get_existing_file_ids()
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
        return

    # 4. Process and upload missing files
    total_files = len(missing_files)
    for i, f in enumerate(missing_files):
        msg = f"Processing {f['name']} ({i+1}/{total_files})"
        logging.info(msg)
        if progress_callback:
            progress_callback(i+1, total_files, msg)
            
        try:
            content = read_file_content(f['id'])
            df_temp = dvw_rallies_to_df(content)
            
            if df_temp.empty:
                logging.warning(f"No rallies found in {f['name']}")
                continue

            # Add file metadata
            df_temp['file_id'] = f['id']
            df_temp['file_name'] = f['name']
            
            # --- Apply Manual Fixes (Reggio Emilia) ---
            TEAM_NAME_TO_FIX = "Conad Reggio Emilia"
            # We need to be careful here. If we process file by file, we might not see all IDs 
            # to pick a "canonical" one globally. 
            # However, usually the fix is to map specific known bad IDs to a good one.
            # Or we can just do a local fix if we see multiple IDs in this single file (unlikely).
            # The original code scanned ALL data to find the set of IDs.
            # Here we are processing one file at a time.
            # If we want to be safe, we might need a hardcoded mapping if we know the IDs.
            # For now, let's skip the complex dynamic fix and assume the user might run a cleanup later
            # OR we can try to apply it if we see the name.
            # 
            # Let's just proceed with raw data for now, as the dynamic fix requires global context.
            # Or we can implement a simpler version: if team name is X, force ID to Y?
            # But we don't know Y without looking at other files.
            # 
            # Wait, the user's previous code did:
            # affected_home = all_data[all_data["team_h"] == TEAM_NAME_TO_FIX]["team_id_h"].unique()
            # ...
            # canonical_id = str(min(all_affected_ids))
            # 
            # Since we are uploading incrementally, we might introduce inconsistency if we don't fix it.
            # BUT, if we read from DB later, we can fix it on the full dataset.
            # 
            # Let's add the match_alternative_id which is required by the schema.
            df_temp["match_alternative_id"] = (
                df_temp["match_date"].astype(str)
                + " | "
                + df_temp["team_id_h"].astype(str)
                + " | "
                + df_temp["team_id_a"].astype(str)
            )
            
            # Upload to DB
            upload_rallies(df_temp)
            logging.info(f"Uploaded {len(df_temp)} rallies for {f['name']}")
            
        except Exception as e:
            logging.error(f"Error processing/uploading file {f['name']}: {e}")

def load_data_from_db() -> pd.DataFrame:
    """
    Loads the full dataset from Supabase.
    """
    logging.info("Fetching data from Supabase...")
    df = fetch_all_rallies()
    
    if not df.empty:
        # Sort as requested
        # We need to ensure columns are correct types
        # rally_idx is int
        if 'rally_idx' in df.columns:
            df['rally_idx'] = pd.to_numeric(df['rally_idx'], errors='coerce').fillna(0).astype(int)
            
        sort_cols = ['match_date', 'file_id', 'rally_idx']
        # Check if cols exist
        existing_sort_cols = [c for c in sort_cols if c in df.columns]
        if existing_sort_cols:
            df = df.sort_values(by=existing_sort_cols).reset_index(drop=True)
            
        # Apply the Reggio Emilia fix globally here, after fetching all data
        TEAM_NAME_TO_FIX = "Conad Reggio Emilia"
        if "team_h" in df.columns and "team_a" in df.columns:
            affected_home = df[df["team_h"] == TEAM_NAME_TO_FIX]["team_id_h"].unique()
            affected_away = df[df["team_a"] == TEAM_NAME_TO_FIX]["team_id_a"].unique()
            all_affected_ids = list(set(list(affected_home) + list(affected_away)))
            
            if len(all_affected_ids) > 1:
                canonical_id = str(min(all_affected_ids))
                logging.warning(f"Applying fix for {TEAM_NAME_TO_FIX}: {all_affected_ids} -> {canonical_id}")
                for old_id in all_affected_ids:
                    if old_id != canonical_id:
                        df.loc[df["team_id_h"] == old_id, "team_id_h"] = canonical_id
                        df.loc[df["team_id_a"] == old_id, "team_id_a"] = canonical_id
                        
                # Re-generate match_alternative_id if needed? 
                # The DB has the old one. We might want to update it in memory for analysis consistency.
                df["match_alternative_id"] = (
                    df["match_date"].astype(str)
                    + " | "
                    + df["team_id_h"].astype(str)
                    + " | "
                    + df["team_id_a"].astype(str)
                )

    return df

if __name__ == "__main__":
    
    output_path = './clean_data/clean_data.csv'
    
    # Check if GDrive is configured in secrets
    folder_ids = []
    if "gdrive" in st.secrets:
        if "folder_ids" in st.secrets["gdrive"]:
            folder_ids = st.secrets["gdrive"]["folder_ids"]
        elif "folder_id" in st.secrets["gdrive"]:
            folder_ids = [st.secrets["gdrive"]["folder_id"]]

    if folder_ids:
        # 1. Update Database with any new files
        update_database(folder_ids)
        
        # 2. Load all data from Database
        all_data = load_data_from_db()
        
        if not all_data.empty:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logging.info('Saving file : {}'.format(output_path))
            all_data.to_csv(output_path, index=False)
            print(f"Loaded {len(all_data)} rows.")
        else:
            logging.warning("No data found in database.")
    else:
        logging.error("No GDrive folder IDs found in secrets.")
