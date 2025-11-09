import os
import json
import pandas as pd
from datetime import datetime

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_files_sorted(dir_path):
    """Return a sorted list of filenames in a directory."""
    return sorted(
        [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    )

import pandas as pd
import re

import re
import pandas as pd

import re
import pandas as pd


def dvw_rallies_to_df(path: str) -> pd.DataFrame:
    """
    Read a Data Volley DVW-like text file (CP1252) and return 1 row per rally.

    Output columns:
        match_type, match_date,
        team_id_h, team_id_a, team_h, team_a,
        set_number,
        pre_set_won_h, pre_set_won_a,
        pre_point_won_h, pre_point_won_a,
        p_h, p_a,
        post_set_won_h, post_set_won_a,
        post_point_won_h, post_point_won_a,
        point_won_h, point_won_a, point_won_team,
        serve_h, serve_a, serve_team
    """
    # -------------------------------------------------------------------------
    # 1) read file
    # -------------------------------------------------------------------------
    with open(path, "r", encoding="cp1252", errors="ignore") as f:
        lines = f.read().splitlines()

    match_date = None
    match_type = None
    team_id_h = None
    team_h = None
    team_id_a = None
    team_a = None

    # -------------------------------------------------------------------------
    # 2) parse header blocks robustly (your style: tag on a line, data after)
    # -------------------------------------------------------------------------
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()

        # ----- MATCH -----
        if line == "[3MATCH]":
            # take the next non-empty, non-tag line as the main match line
            j = i + 1
            while j < n and (not lines[j].strip() or lines[j].strip().startswith("[")):
                # rare case: empty line right after, skip
                j += 1
            if j < n:
                match_line = lines[j].strip()
                parts = [p.strip() for p in match_line.split(";")]
                # example:
                # 0 date  -> 08/10/2025
                # 1 time  -> 20.30.00
                # 2 season
                # 3 competition -> Regular Season ...
                # 4 match type  -> Amichevole
                if len(parts) > 0:
                    match_date = parts[0]  # "08/10/2025"
                if len(parts) > 4:
                    match_type = parts[4]  # "Amichevole"
            i = j  # continue from here
        # ----- TEAMS -----
        elif line == "[3TEAMS]":
            # next line = home team
            # next line after that = away team
            if i + 1 < n:
                home_line = lines[i + 1].strip()
                home_parts = [p.strip() for p in home_line.split(";")]
                if len(home_parts) >= 2:
                    team_id_h = home_parts[0]
                    team_h = home_parts[1]
            if i + 2 < n:
                away_line = lines[i + 2].strip()
                # make sure it's not a new tag
                if not away_line.startswith("["):
                    away_parts = [p.strip() for p in away_line.split(";")]
                    if len(away_parts) >= 2:
                        team_id_a = away_parts[0]
                        team_a = away_parts[1]
            # skip ahead
            i += 2
        i += 1

    # -------------------------------------------------------------------------
    # 3) walk through scout events
    # -------------------------------------------------------------------------
    rows = []

    inside_scout = False

    current_set = 1
    sets_h = 0
    sets_a = 0
    pts_h = 0
    pts_a = 0

    # current setter positions BEFORE the next rally
    home_setter_pos = None
    away_setter_pos = None

    # serving team for the upcoming rally
    current_server_team = None  # 'h' or 'a'

    last_rally_idx = None  # to patch post_set_won_* at end of set

    def process_event(ev: str):
        nonlocal current_set, sets_h, sets_a, pts_h, pts_a
        nonlocal home_setter_pos, away_setter_pos
        nonlocal current_server_team, last_rally_idx

        ev = ev.strip()
        if not ev:
            return

        # -------------------------------------------------------------
        # set end marker
        # -------------------------------------------------------------
        m_endset = re.match(r"^\*\*(\d+)set", ev, flags=re.IGNORECASE)
        if m_endset:
            # finalize this set on the last rally
            if last_rally_idx is not None:
                if pts_h > pts_a:
                    sets_h += 1
                elif pts_a > pts_h:
                    sets_a += 1
                rows[last_rally_idx]["post_set_won_h"] = sets_h
                rows[last_rally_idx]["post_set_won_a"] = sets_a

            # move to next set
            current_set = int(m_endset.group(1)) + 1
            pts_h = 0
            pts_a = 0
            home_setter_pos = None
            away_setter_pos = None
            current_server_team = None
            last_rally_idx = None
            return

        # -------------------------------------------------------------
        # setter positions (home)
        #  *z6...
        #  *z6>LUp...
        # -------------------------------------------------------------
        m_home_z = re.match(r"^\*z([1-6])", ev)
        if m_home_z:
            home_setter_pos = int(m_home_z.group(1))
            return

        # -------------------------------------------------------------
        # setter positions (away)
        #  az5...
        #  az5>LUp...
        # -------------------------------------------------------------
        m_away_z = re.match(r"^az([1-6])", ev)
        if m_away_z:
            away_setter_pos = int(m_away_z.group(1))
            return

        # -------------------------------------------------------------
        # serving detection (must come before the rally point line)
        #   *06S..., *10SQ..., a08SM..., a02SQ..., etc.
        # -------------------------------------------------------------
        if re.match(r"^\*\d+S", ev) or re.match(r"^\*\d+SQ", ev):
            current_server_team = "h"
            return
        if re.match(r"^a\d+S", ev) or re.match(r"^a\d+SQ", ev):
            current_server_team = "a"
            return

        # -------------------------------------------------------------
        # scoreboard / rally lines
        # -------------------------------------------------------------
        m_home_p = re.match(r"^\*p(\d+):(\d+)", ev)
        m_away_p = re.match(r"^ap(\d+):(\d+)", ev)

        if not m_home_p and not m_away_p:
            # not a rally, ignore
            return

        if m_home_p:
            new_h = int(m_home_p.group(1))
            new_a = int(m_home_p.group(2))
        else:
            new_h = int(m_away_p.group(1))
            new_a = int(m_away_p.group(2))

        # pre
        pre_point_h = pts_h
        pre_point_a = pts_a
        pre_set_h = sets_h
        pre_set_a = sets_a

        # who won
        if new_h > pts_h:
            winner = "h"
        elif new_a > pts_a:
            winner = "a"
        else:
            winner = "h" if ev.startswith("*p") else "a"

        # serving team for this rally
        if current_server_team is None:
            serving_team = winner  # fallback
        else:
            serving_team = current_server_team

        serve_h = 1 if serving_team == "h" else 0
        serve_a = 1 if serving_team == "a" else 0

        row = {
            "match_type": match_type,
            "match_date": match_date,
            "team_id_h": team_id_h,
            "team_id_a": team_id_a,
            "team_h": team_h,
            "team_a": team_a,
            "set_number": current_set,
            "pre_set_won_h": pre_set_h,
            "pre_set_won_a": pre_set_a,
            "pre_point_won_h": pre_point_h,
            "pre_point_won_a": pre_point_a,
            "p_h": home_setter_pos,
            "p_a": away_setter_pos,
            "post_set_won_h": sets_h,
            "post_set_won_a": sets_a,
            "post_point_won_h": new_h,
            "post_point_won_a": new_a,
            "point_won_h": 1 if winner == "h" else 0,
            "point_won_a": 1 if winner == "a" else 0,
            "point_won_team": winner,
            "serve_h": serve_h,
            "serve_a": serve_a,
            "serve_team": serving_team,
        }
        rows.append(row)
        last_rally_idx = len(rows) - 1

        # update running score
        pts_h = new_h
        pts_a = new_a

        # after a point, DV will tell us new z-lines, so we forget serve
        current_server_team = None

    # -------------------------------------------------------------------------
    # 4) run through file and feed events
    # -------------------------------------------------------------------------
    for line in lines:
        if "[3SCOUT]" in line:
            inside_scout = True
            # may have events on same line
            after = line.split("[3SCOUT]", 1)[1].strip()
            if after:
                for ev in after.split():
                    process_event(ev)
            continue

        if not inside_scout:
            continue

        # stop at next section
        if line.strip().startswith("[") and not line.strip().startswith("[3SCOUT]"):
            break

        for ev in line.strip().split():
            process_event(ev)

    # -------------------------------------------------------------------------
    # 5) make dataframe
    # -------------------------------------------------------------------------
    return pd.DataFrame(rows)



if __name__ == "__main__":

    input_dir_path = "./data"
    output_path = './clean_data.csv'

    # list files in the directory
    input_file_list = list_files_sorted(input_dir_path)
    logging.info(input_file_list)

    all_data = []

    for i, fn in enumerate(input_file_list):
        logging.info("Processing file {}: {}/{}".format(fn, i+1, len(input_file_list)))
        df_temp = dvw_rallies_to_df(fn)
        all_data.append(df_temp)

    all_data = pd.concat(all_data)
    logging.info('Saving file : {}'.format(output_path))
    all_data.to_csv(output_path, index=False)
    print(all_data)
