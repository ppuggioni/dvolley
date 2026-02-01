import logging

import pandas as pd

from dvolley.services.database_connection import supabase
from dvolley.data.normalization import normalize_date_str
from dvolley.services.db import fetch_all_rallies, fetch_all_touches

logger = logging.getLogger(__name__)


def normalize_dates(only_match_alt: bool = False, dry_run: bool = False) -> None:
    update_date = not only_match_alt
    update_match_alt = True

    rally_updated = _update_rally_dates(
        dry_run=dry_run,
        update_date=update_date,
        update_match_alt=update_match_alt,
    )
    touch_updated = _update_touch_dates(
        dry_run=dry_run,
        update_date=update_date,
        update_match_alt=update_match_alt,
    )

    if dry_run:
        logger.info("Dry run complete. Rally updates: %s, Touch updates: %s", rally_updated, touch_updated)
    else:
        logger.info("Update complete. Rally updates: %s, Touch updates: %s", rally_updated, touch_updated)


def _update_rally_dates(
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

        (
            supabase.table("rally_level_data")
            .update(update_payload)
            .eq("file_id", file_id)
            .eq("rally_idx", rally_idx)
            .execute()
        )
        updated += 1

    return updated


def _update_touch_dates(
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

        (
            supabase.table("touch_level_data")
            .update(update_payload)
            .eq("unique_row_id", unique_row_id)
            .execute()
        )
        updated += 1

    return updated
