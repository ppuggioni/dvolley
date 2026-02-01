import argparse
import logging

import pandas as pd

from analysis_regr import VolleyballBreakpointSideoutRegModelNoHome
from dvolley.services.data_loader import update_database, update_database_full, load_data_from_db
from dvolley.services.db import delete_all_rallies, delete_all_touches
from dvolley.services.maintenance import normalize_dates
from dvolley.services.runtime import get_folder_ids_from_secrets
from dvolley.config import DEFAULT_ALPHA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cmd_reset_db(args: argparse.Namespace) -> None:
    if not args.confirm:
        logger.error("Refusing to run without --confirm (destructive).")
        return

    folder_ids = get_folder_ids_from_secrets()
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


def cmd_normalize_dates(args: argparse.Namespace) -> None:
    normalize_dates(only_match_alt=args.only_match_alt, dry_run=args.dry_run)


def cmd_fit_model(args: argparse.Namespace) -> None:
    df = load_data_from_db()
    if df.empty:
        logger.error("No rally data available. Load data from DB first.")
        return

    model = VolleyballBreakpointSideoutRegModelNoHome(
        half_life_days=90.0,
        alpha=args.alpha,
        max_iter=5000,
        random_state=42,
    )

    required_cols = model.REQUIRED_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error("Rally data missing required columns: %s", missing)
        return

    clean_df = df[required_cols].dropna()
    if "match_date" in clean_df.columns:
        clean_df = clean_df.copy()
        clean_df["match_date"] = pd.to_datetime(clean_df["match_date"], format="mixed", dayfirst=True)

    for col in ["team_id_h", "team_id_a"]:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].astype(str)

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        tmp_path = tmp.name
        clean_df.to_csv(tmp_path, index=False, encoding="utf-8")

    try:
        model.load_data(tmp_path, encoding="utf-8")
        model.fit()
        params_df = model.viz_parameters()
        params_df.to_csv(args.out, index=False)
        logger.info("Saved parameters to %s", args.out)
    finally:
        import os

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="dvolley CLI")
    sub = parser.add_subparsers(dest="command")

    reset = sub.add_parser("reset-db", help="Wipe DB tables and reload from Google Drive.")
    reset.add_argument("--confirm", action="store_true", help="Required to delete existing DB data.")
    reset.set_defaults(func=cmd_reset_db)

    norm = sub.add_parser("normalize-dates", help="Normalize match_date/match_alternative_id in DB.")
    norm.add_argument("--dry-run", action="store_true", help="Count rows but do not update.")
    norm.add_argument("--only-match-alt", action="store_true", help="Update match_alternative_id only.")
    norm.set_defaults(func=cmd_normalize_dates)

    fit = sub.add_parser("fit-model", help="Fit regression model from DB and save parameters to CSV.")
    fit.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Model alpha value.")
    fit.add_argument("--out", default="params_out_break_sideout.csv", help="Output CSV path.")
    fit.set_defaults(func=cmd_fit_model)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
