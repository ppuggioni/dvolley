from __future__ import annotations

import os
import json
import pandas as pd
from datetime import datetime

import datavolley as dv

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_files_sorted(dir_path):
    """Return a sorted list of filenames in a directory."""
    return sorted(
        [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    )

import re
from datetime import datetime, date, time
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np


import pandas as pd
import numpy as np

def add_rally_metadata(
    df,
    set_col="set_number",
    time_col="video_time",
    skill_col="skill",
    team_col="team",
    home_team_col="home_team",
    away_team_col="visiting_team",
    home_score_col="home_team_score",
    away_score_col="visiting_team_score",
    code_col="code",
    sort_by_video_time=False
):
    df = df.copy()
    df["_orig_idx"] = np.arange(len(df))

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[set_col] = pd.to_numeric(df[set_col], errors="coerce").astype("Int64")
    for c in (home_score_col, away_score_col):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if sort_by_video_time:
        df[time_col] = df[time_col].fillna(method='ffill').fillna(0)
        df = df.sort_values([set_col, time_col, "_orig_idx"], kind="mergesort")

    n = len(df)
    rally_numbers = np.zeros(n, dtype=int)
    point_won_by_col = [None] * n

    serve_records = []   # (set, rally, serving_team)
    point_records = []   # (set, rally, point_won_by)

    current_set = None
    current_rally = 0
    last_home_score = None
    last_away_score = None

    for i, row in df.iterrows():
        row_set = row[set_col]

        if current_set is None or row_set != current_set:
            current_set = row_set
            current_rally = 1
            last_home_score = row[home_score_col]
            last_away_score = row[away_score_col]

        rally_numbers[i] = current_rally

        if row[skill_col] == "Serve":
            serve_team = row[team_col] if pd.notna(row[team_col]) and row[team_col] != "" else None
            if serve_team:
                serve_records.append((row_set, current_rally, serve_team))

        if row[skill_col] == "Point":
            winner = None
            if row[home_score_col] > (last_home_score if last_home_score is not None else -1):
                winner = row[home_team_col]
            elif row[away_score_col] > (last_away_score if last_away_score is not None else -1):
                winner = row[away_team_col]
            else:
                code_val = str(row.get(code_col, ""))
                if code_val.startswith("*p"):
                    winner = row[home_team_col]
                elif code_val.startswith("ap"):
                    winner = row[away_team_col]

            point_won_by_col[i] = winner
            point_records.append((row_set, current_rally, winner))

            current_rally += 1

        last_home_score = row[home_score_col]
        last_away_score = row[away_score_col]

    df["rally_number"] = rally_numbers

    # serving team per rally
    if serve_records:
        serve_df = (
            pd.DataFrame(serve_records, columns=[set_col, "rally_number", "serving_team"])
            .dropna(subset=["serving_team"])
            .drop_duplicates([set_col, "rally_number"])
        )
        df = df.merge(serve_df, on=[set_col, "rally_number"], how="left")
    else:
        df["serving_team"] = np.nan

    df["serving_team"] = df.groupby([set_col, "rally_number"])["serving_team"].ffill().bfill()

    # point winner per rally
    if point_records:
        point_df = pd.DataFrame(point_records, columns=[set_col, "rally_number", "point_won_by"])
        point_df = point_df.dropna(subset=["point_won_by"])
        if not point_df.empty:
            point_df = point_df.drop_duplicates([set_col, "rally_number"])
            df = df.merge(point_df, on=[set_col, "rally_number"], how="left", suffixes=("", "_from_point"))
            if "point_won_by_from_point" in df.columns:
                df["point_won_by"] = df["point_won_by_from_point"].combine_first(df.get("point_won_by"))
                df.drop(columns=["point_won_by_from_point"], inplace=True)
        else:
            df["point_won_by"] = np.nan
    else:
        df["point_won_by"] = np.nan

    df["point_won_by"] = df.groupby([set_col, "rally_number"])["point_won_by"].ffill().bfill()

    # receiving team
    df["receiving_team"] = np.where(
        df["serving_team"] == df[home_team_col],
        df[away_team_col],
        np.where(
            df["serving_team"] == df[away_team_col],
            df[home_team_col],
            np.nan,
        ),
    )

    # ===== possessions (updated) =====
    df["possession_number"] = 0
    for (s, r), g in df.groupby([set_col, "rally_number"], sort=False):
        # find first Serve in this rally
        serve_mask = g[skill_col].eq("Serve")
        first_serve_idx = serve_mask.idxmax() if serve_mask.any() else None

        current_pos = 0
        current_team = None
        poss_vals = []

        for idx, row in g.iterrows():
            # before the serve -> stay 0
            if first_serve_idx is not None and idx < first_serve_idx:
                poss_vals.append(0)
                continue

            # after (or at) the serve we start counting
            row_team = row[team_col]

            # if the serve row has no team, fall back to serving_team
            if first_serve_idx is not None and idx == first_serve_idx and (pd.isna(row_team) or row_team == ""):
                row_team = row["serving_team"]

            # empty team even after serve -> keep current possession
            if pd.isna(row_team) or row_team == "":
                poss_vals.append(current_pos)
                continue

            if current_pos == 0:
                current_pos = 1
                current_team = row_team
            else:
                if row_team != current_team:
                    current_pos += 1
                    current_team = row_team
            poss_vals.append(current_pos)

        df.loc[g.index, "possession_number"] = poss_vals
    # ===== end possessions =====

    df = df.sort_values("_orig_idx").drop(columns=["_orig_idx"])
    return df


import re
from datetime import datetime
from typing import Optional, Tuple


def extract_match_date_and_type(path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a Data Volley .dvw-like file, return (match_date_str, match_type_str).

    match_date_str is returned as the original DV string, e.g. "08/10/2025".
    match_type_str is e.g. "Amichevole".

    If something is missing, returns None for that part.
    """
    match_date = None
    match_type = None

    with open(path, "r", encoding="cp1252", errors="ignore") as f:
        lines = [line.strip() for line in f]

    # 1) try to get the line right after [3MATCH]
    match_section_idx = None
    for i, line in enumerate(lines):
        if line.strip().upper() == "[3DATAVOLLEYSCOUT]":
            # not needed, but we know we're in a DV file
            continue
        if line.strip().upper() == "[3MATCH]":
            match_section_idx = i
            break

    if match_section_idx is not None:
        # take the first non-empty line after [3MATCH]
        for j in range(match_section_idx + 1, len(lines)):
            line = lines[j].strip()
            if not line:
                continue
            # this should be the semicolon-separated match line
            parts = line.split(";")
            if parts:
                # date is usually parts[0], like "08/10/2025"
                if len(parts) >= 1 and parts[0].strip():
                    match_date = parts[0].strip()

                # match type is usually the 5th field (index 4)
                if len(parts) >= 5 and parts[4].strip():
                    # clean control chars, just in case
                    mt = parts[4].strip()
                    mt = re.sub(r"[\x00-\x1f]", "", mt)
                    match_type = mt
                break

    # 2) fallback for date: use GENERATOR-DAY if we didn't find the match date
    if match_date is None:
        for line in lines:
            if line.startswith("GENERATOR-DAY:"):
                raw = line.split("GENERATOR-DAY:", 1)[1].strip()
                # DV has "08/10/2025 16:52:35" -> take date part
                match_date = raw.split()[0]
                break

    return match_date, match_type



ORDERED_COLS = [
    # 1) Match identification
    "match_id",
    "match_alternative_id",
    "match_type",
    "match_date",

    # 2) Teams
    "home_team_id",
    "home_team",
    "visiting_team_id",
    "visiting_team",

    # 3) Set and score
    "set_number",
    "home_team_score",
    "visiting_team_score",

    # 4) Rally metadata
    "rally_number",
    "point_won_by",
    "serving_team",
    "receiving_team",
    "setter_position",
    "home_setter_position",
    "visiting_setter_position",
    "possession_number",

    # 5) Event / DV code
    "video_time",
    "code",
    "custom_code",
    "point_phase",
    "attack_phase",

    # 6) Actor
    "team",
    "player_id",
    "player_name",
    "player_number",

    # 7) Skill/action details
    "skill",
    "skill_type",
    "skill_subtype",
    "evaluation_code",

    # 8) Technical context
    "attack_code",
    "set_code",
    "set_type",

    # 9) Zones / counts
    "start_zone",
    "end_zone",
    "end_subzone",
    "num_players_numeric",

    # 10) Coordinates
    "start_coordinate",
    "mid_coordinate",
    "end_coordinate",
    "start_coordinate_x",
    "start_coordinate_y",
    "mid_coordinate_x",
    "mid_coordinate_y",
    "end_coordinate_x",
    "end_coordinate_y",

    # 11) Lineups
    "home_p1", "home_p2", "home_p3", "home_p4", "home_p5", "home_p6",
    "visiting_p1", "visiting_p2", "visiting_p3", "visiting_p4", "visiting_p5", "visiting_p6",
]


def process_dv_file(path: str) -> pd.DataFrame:
    """Read one DV file, enrich, and return a clean DataFrame."""
    raw = dv.read_dv(path)
    df = pd.DataFrame(raw)

    # drop old rally columns if present
    cols_to_exclude = [
        "rally_number", "point_won_by", "serving_team",
        "receiving_team", "possession_number"
    ]
    df = df[[c for c in df.columns if c not in cols_to_exclude]]

    # add rally metadata
    df = add_rally_metadata(df)

    # match-level info
    match_date_str, match_type = extract_match_date_and_type(path)
    if match_date_str:
        match_date_str = pd.to_datetime(match_date_str, dayfirst=True).strftime("%Y-%m-%d")

    df["match_date"] = match_date_str
    df["match_type"] = match_type

    # build alternative id
    df["match_alternative_id"] = (
        df["match_date"].astype(str)
        + " | "
        + df["home_team_id"].astype(str)
        + " | "
        + df["visiting_team_id"].astype(str)
    )

    # order columns (put known ones first)
    existing = [c for c in ORDERED_COLS if c in df.columns]
    extra = [c for c in df.columns if c not in existing]
    df = df[existing + extra]

    return df


def concat_align_and_save(dfs: list[pd.DataFrame], output_path: str) -> pd.DataFrame:
    """
    Take a list of DataFrames, align them to the union of columns,
    concatenate, save to CSV, and return the final DataFrame.
    """
    # union of columns
    all_cols = set()
    for d in dfs:
        all_cols.update(d.columns)
    all_cols = list(all_cols)

    # align
    aligned = [d.reindex(columns=all_cols) for d in dfs]

    # concat
    final_df = pd.concat(aligned, ignore_index=True)

    # save
    final_df.to_csv(output_path, index=False)

    return final_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    input_dir_path = "./data"
    output_path = "./clean_full_data.csv"

    files = list_files_sorted(input_dir_path)
    logging.info("Found %d files", len(files))

    per_file_dfs = []
    for i, fn in enumerate(files):
        logging.info("Processing file %s (%d/%d)", fn, i + 1, len(files))
        df_temp = process_dv_file(fn)
        per_file_dfs.append(df_temp)

    final_df = concat_align_and_save(per_file_dfs, output_path)
    print(final_df)
