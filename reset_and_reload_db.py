import argparse
import logging

import streamlit as st

from db_utils import delete_all_rallies, delete_all_touches
from load_data import update_database
from load_full_data import update_database_full

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_folder_ids() -> list[str]:
    folder_ids: list[str] = []
    if "gdrive" in st.secrets:
        if "folder_ids" in st.secrets["gdrive"]:
            folder_ids = st.secrets["gdrive"]["folder_ids"]
        elif "folder_id" in st.secrets["gdrive"]:
            folder_ids = [st.secrets["gdrive"]["folder_id"]]
    return folder_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wipe Supabase tables and reload all data from Google Drive."
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required to delete existing DB data.",
    )
    args = parser.parse_args()

    if not args.confirm:
        logger.error("Refusing to run without --confirm (destructive).")
        return

    folder_ids = get_folder_ids()
    if not folder_ids:
        logger.error("No Google Drive folder IDs found in secrets.")
        return

    logger.info("Deleting rally_level_data...")
    deleted_files = delete_all_rallies()
    logger.info("Deleted rallies for %s file_ids.", deleted_files)

    logger.info("Deleting touch_level_data...")
    delete_all_touches()
    logger.info("Deleted all touch rows.")

    logger.info("Re-uploading rallies from Google Drive...")
    update_database(folder_ids)

    logger.info("Re-uploading touches from Google Drive...")
    update_database_full(folder_ids)

    logger.info("Reset and reload complete.")


if __name__ == "__main__":
    main()
