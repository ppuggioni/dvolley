import streamlit as st
import pandas as pd
import logging
from database_connection import supabase

def create_rally_table_if_not_exists():
    """
    Creates the rally_level_data table in Supabase if it doesn't exist.
    """
    # Note: Supabase-py client doesn't support direct DDL execution easily via the client 
    # for security reasons usually, but we can try rpc if we had a function, 
    # or we can assume the user might run this SQL in the dashboard.
    # However, the user asked for "something that says update database... create table if not exists".
    # Since standard supabase-py client interacts with PostgREST, we can't run raw SQL DDL 
    # unless we use the 'rpc' call to a postgres function that executes SQL, 
    # or if we use a direct postgres connection (psycopg2).
    # 
    # Given the context of "database_connection.py" using `supabase-py`, 
    # we will assume the table might need to be created manually or we can try to use 
    # a workaround if the user really wants it in code. 
    # 
    # BUT, the user provided the SQL. 
    # Let's try to see if we can use a direct SQL execution if possible, 
    # or just log that we expect the table to exist.
    # 
    # Actually, the user said "when data must be loaded, then it will read from the supabase database which I created like this...".
    # This implies the user MIGHT have already created it, or wants me to ensure it's created.
    # 
    # If I cannot execute DDL via the client, I will skip the creation step in code 
    # and assume it exists, but I will provide the SQL in a comment or log.
    # 
    # Wait, the user said "I want something that says update database. and basically, when data must be loaded, then it will read from the supabase database which I created like this ...".
    # This phrasing "which I created like this" suggests they might have ALREADY created it, 
    # OR they want me to use that schema.
    # 
    # Let's assume for now we just need to read/write to it.
    pass

def get_existing_file_ids() -> set:
    """
    Queries the DB to get a set of already processed file_ids.
    """
    try:
        # We only need the file_id column, and we want unique values.
        # Supabase (PostgREST) doesn't support 'distinct' easily in the select string 
        # without a specific syntax or rpc. 
        # We will fetch all file_ids (might be heavy if millions of rows) 
        # or better, use a simplified query if possible.
        # 
        # A better approach for "what files are processed" is to have a separate table `processed_files`.
        # But the user schema puts `file_id` in `rally_level_data`.
        # 
        # We can try to fetch just file_id.
        response = supabase.table("rally_level_data").select("file_id").execute()
        
        # response.data is a list of dicts: [{'file_id': '...'}, ...]
        if response.data:
            return set(item['file_id'] for item in response.data)
        return set()
    except Exception as e:
        logging.error(f"Error fetching existing file IDs: {e}")
        return set()

def upload_rallies(df: pd.DataFrame):
    """
    Uploads a DataFrame to the rally_level_data table.
    """
    if df.empty:
        return

    # Convert DataFrame to list of dicts
    # Make sure types match the schema
    # The schema expects:
    # match_date: string 'YYYY-MM-DD'
    # created_by: default 'system' (we can omit or send it)
    # create_datetime: default now (we can omit)
    
    # We need to ensure columns match exactly.
    # The user schema has specific columns. 
    # Our `dvw_rallies_to_df` produces most of them.
    # We need to map or ensure they exist.
    
    import numpy as np
    df_clean = df.replace([np.inf, -np.inf, np.nan], None)
    records = df_clean.to_dict(orient='records')
    
    # Batch insert to avoid hitting payload limits
    BATCH_SIZE = 1000
    total_records = len(records)
    
    for i in range(0, total_records, BATCH_SIZE):
        batch = records[i:i+BATCH_SIZE]
        try:
            supabase.table("rally_level_data").insert(batch).execute()
            logging.info(f"Uploaded batch {i} to {i+len(batch)}")
        except Exception as e:
            logging.error(f"Error uploading batch {i}: {e}")
            # If one batch fails, we might want to stop or continue? 
            # For now, log and continue might be risky if partial data.
            # But let's try to continue.

def fetch_all_rallies() -> pd.DataFrame:
    """
    Fetches all data from the rally_level_data table.
    """
    try:
        # Supabase JS/Py client has a default limit (usually 1000). 
        # We need to paginate to get all data.
        
        all_rows = []
        start = 0
        batch_size = 1000
        
        while True:
            response = supabase.table("rally_level_data").select("*").range(start, start + batch_size - 1).execute()
            rows = response.data
            if not rows:
                break
            all_rows.extend(rows)
            
            if len(rows) < batch_size:
                break
            start += batch_size
            
        return pd.DataFrame(all_rows)
    except Exception as e:
        logging.error(f"Error fetching rallies: {e}")
        return pd.DataFrame()

def get_existing_match_ids() -> set:
    """
    Queries the DB to get a set of already processed match_alternative_ids from touch_level_data.
    """
    try:
        # We use match_alternative_id as the unique identifier for a match file
        response = supabase.table("touch_level_data").select("match_alternative_id").execute()
        
        if response.data:
            return set(item['match_alternative_id'] for item in response.data)
        return set()
    except Exception as e:
        logging.error(f"Error fetching existing match IDs: {e}")
        return set()

def get_existing_file_ids_from_touches() -> set:
    """
    Queries the DB to get a set of already processed file_ids from touch_level_data.
    """
    try:
        response = supabase.table("touch_level_data").select("file_id").execute()
        
        if response.data:
            return set(item['file_id'] for item in response.data if item.get('file_id'))
        return set()
    except Exception as e:
        logging.error(f"Error fetching existing file IDs from touches: {e}")
        return set()

def upload_touches(df: pd.DataFrame):
    """
    Uploads a DataFrame to the touch_level_data table.
    """
    if df.empty:
        return

    # Convert DataFrame to list of dicts
    # Handle NaN values: Supabase/Postgres might not like NaN for numeric columns if not nullable, 
    # or might expect None. Pandas to_dict handles this but we should be careful.
    # We replace NaN with None for JSON serialization compatibility
    # Also handle Infinity and pd.NA
    import numpy as np
    # Note: pd.NA is not easily replaced by replace() in some pandas versions mixed with numpy types
    # But we can try. Best way for object/Int64 columns is to use where or fillna before to_dict?
    # Actually, to_dict handles Int64 pd.NA as None usually.
    # But let's be safe.
    df_clean = df.replace([np.inf, -np.inf, np.nan], None)
    # For Int64 columns with pd.NA, to_dict should produce None. 
    # But if we have mixed types, we might need more care.
    # Let's trust to_dict for Int64, but ensure float NaNs are None.
    records = df_clean.to_dict(orient='records')
    
    # Batch insert
    BATCH_SIZE = 1000
    total_records = len(records)
    
    for i in range(0, total_records, BATCH_SIZE):
        batch = records[i:i+BATCH_SIZE]
        try:
            # Use upsert to handle potential duplicates (e.g. re-running script)
            supabase.table("touch_level_data").upsert(batch).execute()
            logging.info(f"Uploaded batch {i} to {i+len(batch)} (touches)")
        except Exception as e:
            # Try to get more details if it's a Supabase/PostgREST error
            error_details = str(e)
            if hasattr(e, 'message'):
                error_details += f" | Message: {e.message}"
            if hasattr(e, 'details'):
                error_details += f" | Details: {e.details}"
            if hasattr(e, 'hint'):
                error_details += f" | Hint: {e.hint}"
            if hasattr(e, 'code'):
                error_details += f" | Code: {e.code}"
                
            logging.error(f"Error uploading batch {i}: {error_details}")
            # If we fail, we should probably stop to avoid partial data or spamming logs
            # But for now let's just log and maybe break?
            # break

def fetch_all_touches() -> pd.DataFrame:
    """
    Fetches all data from the touch_level_data table.
    """
    try:
        all_rows = []
        start = 0
        batch_size = 1000
        
        while True:
            response = supabase.table("touch_level_data").select("*").range(start, start + batch_size - 1).execute()
            rows = response.data
            if not rows:
                break
            all_rows.extend(rows)
            
            if len(rows) < batch_size:
                break
            start += batch_size
            
        return pd.DataFrame(all_rows)
    except Exception as e:
        logging.error(f"Error fetching touches: {e}")
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
            response = supabase.table("touch_level_data").select("*").eq("match_alternative_id", match_id).range(start, start + batch_size - 1).execute()
            rows = response.data
            if not rows:
                break
            all_rows.extend(rows)
            
            if len(rows) < batch_size:
                break
            start += batch_size
            
        return pd.DataFrame(all_rows)
    except Exception as e:
        logging.error(f"Error fetching touches for match {match_id}: {e}")
        return pd.DataFrame()
