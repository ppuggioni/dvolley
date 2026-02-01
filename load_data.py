import os

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import streamlit as st

from dvolley.services.data_loader import load_data_from_db, update_database


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
