from __future__ import annotations

import logging

import pandas as pd

from dvolley.services.database_connection import supabase

logger = logging.getLogger(__name__)


def create_rally_table_if_not_exists():
    """
    Placeholder for DDL creation. Supabase PostgREST doesn't run DDL directly.
    """
    pass


def get_existing_file_ids() -> set:
    """
    Queries the DB to get a set of already processed file_ids.
    """
    try:
        response = supabase.table("rally_level_data").select("file_id").execute()
        if response.data:
            return set(item["file_id"] for item in response.data)
        return set()
    except Exception as e:
        logger.error("Error fetching existing file IDs: %s", e)
        return set()


def upload_rallies(df: pd.DataFrame):
    """
    Uploads a DataFrame to the rally_level_data table.
    """
    if df.empty:
        return

    import numpy as np

    df_clean = df.replace([np.inf, -np.inf, np.nan], None)
    records = df_clean.to_dict(orient="records")

    batch_size = 1000
    total_records = len(records)

    for i in range(0, total_records, batch_size):
        batch = records[i : i + batch_size]
        try:
            supabase.table("rally_level_data").insert(batch).execute()
            logger.info("Uploaded batch %s to %s", i, i + len(batch))
        except Exception as e:
            logger.error("Error uploading batch %s: %s", i, e)


def fetch_all_rallies() -> pd.DataFrame:
    """
    Fetches all data from the rally_level_data table.
    """
    try:
        all_rows = []
        start = 0
        batch_size = 1000

        while True:
            response = (
                supabase.table("rally_level_data")
                .select("*")
                .range(start, start + batch_size - 1)
                .execute()
            )
            rows = response.data
            if not rows:
                break
            all_rows.extend(rows)

            if len(rows) < batch_size:
                break
            start += batch_size

        return pd.DataFrame(all_rows)
    except Exception as e:
        logger.error("Error fetching rallies: %s", e)
        return pd.DataFrame()


def get_existing_match_ids() -> set:
    """
    Queries the DB to get a set of already processed match_alternative_ids from touch_level_data.
    """
    try:
        response = supabase.table("touch_level_data").select("match_alternative_id").execute()
        if response.data:
            return set(item["match_alternative_id"] for item in response.data)
        return set()
    except Exception as e:
        logger.error("Error fetching existing match IDs: %s", e)
        return set()


def get_existing_file_ids_from_touches() -> set:
    """
    Queries the DB to get a set of already processed file_ids from touch_level_data.
    """
    try:
        response = supabase.table("touch_level_data").select("file_id").execute()
        if response.data:
            return set(item["file_id"] for item in response.data if item.get("file_id"))
        return set()
    except Exception as e:
        logger.error("Error fetching existing file IDs from touches: %s", e)
        return set()


def upload_touches(df: pd.DataFrame):
    """
    Uploads a DataFrame to the touch_level_data table.
    """
    if df.empty:
        return

    import numpy as np

    df_clean = df.replace([np.inf, -np.inf, np.nan], None)
    records = df_clean.to_dict(orient="records")

    batch_size = 1000
    total_records = len(records)

    for i in range(0, total_records, batch_size):
        batch = records[i : i + batch_size]
        try:
            supabase.table("touch_level_data").upsert(batch).execute()
            logger.info("Uploaded batch %s to %s (touches)", i, i + len(batch))
        except Exception as e:
            error_details = str(e)
            if hasattr(e, "message"):
                error_details += f" | Message: {e.message}"
            if hasattr(e, "details"):
                error_details += f" | Details: {e.details}"
            if hasattr(e, "hint"):
                error_details += f" | Hint: {e.hint}"
            if hasattr(e, "code"):
                error_details += f" | Code: {e.code}"

            logger.error("Error uploading batch %s: %s", i, error_details)


def fetch_all_touches() -> pd.DataFrame:
    """
    Fetches all data from the touch_level_data table.
    """
    try:
        all_rows = []
        start = 0
        batch_size = 1000

        while True:
            response = (
                supabase.table("touch_level_data")
                .select("*")
                .range(start, start + batch_size - 1)
                .execute()
            )
            rows = response.data
            if not rows:
                break
            all_rows.extend(rows)

            if len(rows) < batch_size:
                break
            start += batch_size

        return pd.DataFrame(all_rows)
    except Exception as e:
        logger.error("Error fetching touches: %s", e)
        return pd.DataFrame()


def fetch_touches_by_match_id(match_id: str) -> pd.DataFrame:
    """
    Fetches all data for a specific match from the touch_level_data table.
    """
    try:
        all_rows = []
        start = 0
        batch_size = 1000

        while True:
            response = (
                supabase.table("touch_level_data")
                .select("*")
                .eq("match_alternative_id", match_id)
                .range(start, start + batch_size - 1)
                .execute()
            )
            rows = response.data
            if not rows:
                break
            all_rows.extend(rows)

            if len(rows) < batch_size:
                break
            start += batch_size

        return pd.DataFrame(all_rows)
    except Exception as e:
        logger.error("Error fetching touches for match %s: %s", match_id, e)
        return pd.DataFrame()


def fetch_touches_by_match_ids(match_ids: list[str]) -> pd.DataFrame:
    """
    Fetches all touch rows for a list of match_alternative_id values.
    """
    if not match_ids:
        return pd.DataFrame()

    try:
        all_rows = []
        page_size = 1000
        chunk_size = 20
        clean_ids = [str(m) for m in match_ids if m]

        for i in range(0, len(clean_ids), chunk_size):
            chunk = clean_ids[i : i + chunk_size]
            start = 0

            while True:
                response = (
                    supabase.table("touch_level_data")
                    .select("*")
                    .in_("match_alternative_id", chunk)
                    .range(start, start + page_size - 1)
                    .execute()
                )
                rows = response.data
                if not rows:
                    break
                all_rows.extend(rows)

                if len(rows) < page_size:
                    break
                start += page_size

        return pd.DataFrame(all_rows)
    except Exception as e:
        logger.error("Error fetching touches for match IDs: %s", e)
        return pd.DataFrame()


def delete_all_rallies() -> int:
    """
    Deletes all rows from rally_level_data by file_id batches.
    Returns the number of file_ids processed.
    """
    file_ids = get_existing_file_ids()
    for file_id in file_ids:
        try:
            supabase.table("rally_level_data").delete().eq("file_id", file_id).execute()
        except Exception as e:
            logger.error("Error deleting rallies for file_id %s: %s", file_id, e)
    return len(file_ids)


def delete_all_touches() -> None:
    """
    Deletes all rows from touch_level_data using unique_row_id filter.
    """
    try:
        supabase.table("touch_level_data").delete().gte("unique_row_id", 0).execute()
    except Exception as e:
        logger.error("Error deleting touches: %s", e)
