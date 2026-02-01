import argparse
import logging
from typing import Optional

import pandas as pd

from database_connection import supabase
from db_utils import fetch_all_rallies, fetch_all_touches

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_date_str(date_val) -> Optional[str]:
    if date_val is None or pd.isna(date_val):
        return None
    date_str = str(date_val).strip()
    if not date_str:
        return None
    try:
        return pd.to_datetime(date_str, errors="coerce", dayfirst=True).date().isoformat()
    except Exception:
        return date_str


def update_rally_dates(
    dry_run: bool = False,
    update_date: bool = True,
    update_match_alt: bool = True,
) -> int:
    df = fetch_all_rallies()
    if df.empty or "match_date" not in df.columns:
        logger.info("No rally data found or missing match_date column.")
        return 0

    df = df.copy()
    df["match_date_norm"] = df["match_date"].apply(normalize_date_str)
    df["match_date_str"] = df["match_date"].astype(str)
    df["match_date_norm_str"] = df["match_date_norm"].astype(str)
    if "match_alternative_id" in df.columns:
        df["match_alt_str"] = df["match_alternative_id"].astype(str)
    else:
        df["match_alt_str"] = None

    if "team_id_h" in df.columns and "team_id_a" in df.columns:
        df["match_alt_norm"] = (
            df["match_date_norm"].astype(str)
            + " | "
            + df["team_id_h"].astype(str)
            + " | "
            + df["team_id_a"].astype(str)
        )
    else:
        df["match_alt_norm"] = None
    to_fix = df[df["match_date_str"] != df["match_date_norm_str"]]
    to_fix_alt = df[
        (df["match_alt_norm"].notna())
        & (df["match_alt_str"].notna())
        & (df["match_alt_str"] != df["match_alt_norm"].astype(str))
    ]

    logger.info("Rally rows needing update: %s", len(to_fix))
    logger.info("Rally rows needing match_alternative_id update: %s", len(to_fix_alt))
    updated = 0

    # Update match_date (and match_alternative_id if needed) using file_id + rally_idx
    for _, row in df.iterrows():
        file_id = row.get("file_id")
        rally_idx = row.get("rally_idx")
        new_date = row.get("match_date_norm")
        old_date = row.get("match_date")
        new_match_alt = row.get("match_alt_norm")
        old_match_alt = row.get("match_alternative_id")

        if file_id is None or pd.isna(file_id) or new_date is None:
            continue

        if rally_idx is None or pd.isna(rally_idx):
            continue

        try:
            rally_idx = int(rally_idx)
        except Exception:
            continue

        needs_date = update_date and str(old_date) != str(new_date)
        needs_alt = update_match_alt and new_match_alt is not None and str(old_match_alt) != str(new_match_alt)
        if not needs_date and not needs_alt:
            continue

        if dry_run:
            updated += 1
            continue

        update_payload = {}
        if needs_date:
            update_payload["match_date"] = new_date
        if needs_alt:
            update_payload["match_alternative_id"] = new_match_alt

        query = (
            supabase.table("rally_level_data")
            .update(update_payload)
            .eq("file_id", file_id)
            .eq("rally_idx", rally_idx)
        )
        query.execute()
        updated += 1

    return updated


def update_touch_dates(
    dry_run: bool = False,
    update_date: bool = True,
    update_match_alt: bool = True,
) -> int:
    df = fetch_all_touches()
    if df.empty or "match_date" not in df.columns:
        logger.info("No touch data found or missing match_date column.")
        return 0

    df = df.copy()
    df["match_date_norm"] = df["match_date"].apply(normalize_date_str)
    df["match_date_str"] = df["match_date"].astype(str)
    df["match_date_norm_str"] = df["match_date_norm"].astype(str)
    if "match_alternative_id" in df.columns:
        df["match_alt_str"] = df["match_alternative_id"].astype(str)
    else:
        df["match_alt_str"] = None

    if "home_team_id" in df.columns and "visiting_team_id" in df.columns:
        df["match_alt_norm"] = (
            df["match_date_norm"].astype(str)
            + " | "
            + df["home_team_id"].astype(str)
            + " | "
            + df["visiting_team_id"].astype(str)
        )
    else:
        df["match_alt_norm"] = None
    to_fix = df[df["match_date_str"] != df["match_date_norm_str"]]
    to_fix_alt = df[
        (df["match_alt_norm"].notna())
        & (df["match_alt_str"].notna())
        & (df["match_alt_str"] != df["match_alt_norm"].astype(str))
    ]

    logger.info("Touch rows needing update: %s", len(to_fix))
    logger.info("Touch rows needing match_alternative_id update: %s", len(to_fix_alt))
    updated = 0

    # Update match_date (and match_alternative_id if needed) using unique_row_id
    for _, row in df.iterrows():
        unique_row_id = row.get("unique_row_id")
        new_date = row.get("match_date_norm")
        old_date = row.get("match_date")
        new_match_alt = row.get("match_alt_norm")
        old_match_alt = row.get("match_alternative_id")

        if unique_row_id is None or pd.isna(unique_row_id) or new_date is None:
            continue

        try:
            unique_row_id = int(unique_row_id)
        except Exception:
            continue

        needs_date = update_date and str(old_date) != str(new_date)
        needs_alt = update_match_alt and new_match_alt is not None and str(old_match_alt) != str(new_match_alt)
        if not needs_date and not needs_alt:
            continue

        if dry_run:
            updated += 1
            continue

        update_payload = {}
        if needs_date:
            update_payload["match_date"] = new_date
        if needs_alt:
            update_payload["match_alternative_id"] = new_match_alt

        query = (
            supabase.table("touch_level_data")
            .update(update_payload)
            .eq("unique_row_id", unique_row_id)
        )
        query.execute()
        updated += 1

    return updated


def main():
    parser = argparse.ArgumentParser(description="Normalize DB match_date to YYYY-MM-DD.")
    parser.add_argument("--dry-run", action="store_true", help="Count rows but do not update.")
    parser.add_argument(
        "--only-match-alt",
        action="store_true",
        help="Update match_alternative_id only (skip match_date updates).",
    )
    args = parser.parse_args()

    update_date = not args.only_match_alt
    update_match_alt = True

    rally_updated = update_rally_dates(
        dry_run=args.dry_run,
        update_date=update_date,
        update_match_alt=update_match_alt,
    )
    touch_updated = update_touch_dates(
        dry_run=args.dry_run,
        update_date=update_date,
        update_match_alt=update_match_alt,
    )

    if args.dry_run:
        logger.info("Dry run complete. Rally updates: %s, Touch updates: %s", rally_updated, touch_updated)
    else:
        logger.info("Update complete. Rally updates: %s, Touch updates: %s", rally_updated, touch_updated)


if __name__ == "__main__":
    main()
