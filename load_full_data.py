from __future__ import annotations

import os

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import streamlit as st

from dvolley.services.data_loader import (
    load_full_data_from_db,
    load_match_data_from_db,
    update_database_full,
)
from dvolley.data.full_parser import concat_align_and_save, list_files_sorted, process_dv_file_content

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
