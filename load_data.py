import os
import json
import pandas as pd

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import streamlit as st

from db_utils import fetch_all_rallies, get_existing_file_ids, upload_rallies
from dvolley.data.dvw_parser import dvw_rallies_to_df
from dvolley.data.normalization import normalize_date_str
from gdrive_utils import list_files_in_folder, read_file_content


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
            # Normalize date format to YYYY-MM-DD
            if "match_date" in df_temp.columns:
                df_temp["match_date"] = df_temp["match_date"].apply(normalize_date_str)
            
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
        # Normalize date format to YYYY-MM-DD
        if "match_date" in df.columns:
            df["match_date"] = df["match_date"].apply(normalize_date_str)
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
