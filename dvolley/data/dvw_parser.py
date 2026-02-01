from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

from .normalization import normalize_date_str
from dvolley.config import DATE_FORMATS


def dvw_rallies_to_df(file_content: str) -> pd.DataFrame:
    """
    Read a Data Volley DVW-like text file content (already decoded) and return 1 row per rally.
    """
    # -------------------------------------------------------------------------
    # 1) read file content
    # -------------------------------------------------------------------------
    lines = file_content.splitlines()

    match_date = None
    match_type = "Unknown"  # Default value
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
                    raw_date = parts[0]  # "08/10/2025"
                    match_date = raw_date
                    for fmt in DATE_FORMATS:
                        try:
                            match_date = datetime.strptime(raw_date, fmt).strftime("%Y-%m-%d")
                            break
                        except ValueError:
                            continue
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
                    team_id_h = str(home_parts[0])  # Always string
                    team_h = home_parts[1]
            if i + 2 < n:
                away_line = lines[i + 2].strip()
                # make sure it's not a new tag
                if not away_line.startswith("["):
                    away_parts = [p.strip() for p in away_line.split(";")]
                    if len(away_parts) >= 2:
                        team_id_a = str(away_parts[0])  # Always string
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
    home_setter_pos = 0  # Default to 0
    away_setter_pos = 0  # Default to 0

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
            home_setter_pos = 0
            away_setter_pos = 0
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
            "rally_idx": len(rows),
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
    df = pd.DataFrame(rows)
    if not df.empty and "match_date" in df.columns:
        df["match_date"] = df["match_date"].apply(normalize_date_str)
    return df
