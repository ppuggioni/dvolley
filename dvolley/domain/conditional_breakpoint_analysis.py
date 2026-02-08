from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


ATTACK_QUALITY_CODES = ["#", "+", "!", "-", "/", "="]


@dataclass
class ConditionalBreakpointResult:
    rally_df: pd.DataFrame
    quality_summary: pd.DataFrame
    rotation_quality_summary: pd.DataFrame
    rotation_probability_pivot: pd.DataFrame
    player_summary: pd.DataFrame
    player_quality_summary: pd.DataFrame
    diagnostics: dict[str, int]
    mode: str
    rotation_axis_label: str


def build_conditional_breakpoint_analysis(
    touches_df: pd.DataFrame,
    team_id: str,
    mode: str,
    selected_match_ids: Optional[list[str]] = None,
) -> ConditionalBreakpointResult:
    mode = mode.strip().lower()
    if mode not in {"sideout", "breakpoint"}:
        raise ValueError("mode must be 'sideout' or 'breakpoint'")

    rally_df, diagnostics = _build_rally_df(touches_df, team_id, mode, selected_match_ids)
    rotation_axis_label = (
        "Serving rotation (selected team)"
        if mode == "breakpoint"
        else "Receiving rotation (selected team)"
    )

    if rally_df.empty:
        empty = pd.DataFrame()
        return ConditionalBreakpointResult(
            rally_df=empty,
            quality_summary=empty,
            rotation_quality_summary=empty,
            rotation_probability_pivot=empty,
            player_summary=empty,
            player_quality_summary=empty,
            diagnostics=diagnostics,
            mode=mode,
            rotation_axis_label=rotation_axis_label,
        )

    quality_summary = _build_quality_summary(rally_df)
    rotation_quality_summary = _build_rotation_quality_summary(rally_df)
    rotation_probability_pivot = _build_rotation_probability_pivot(rotation_quality_summary)
    player_summary, player_quality_summary = _build_player_tables(rally_df, mode)

    return ConditionalBreakpointResult(
        rally_df=rally_df,
        quality_summary=quality_summary,
        rotation_quality_summary=rotation_quality_summary,
        rotation_probability_pivot=rotation_probability_pivot,
        player_summary=player_summary,
        player_quality_summary=player_quality_summary,
        diagnostics=diagnostics,
        mode=mode,
        rotation_axis_label=rotation_axis_label,
    )


def _build_rally_df(
    touches_df: pd.DataFrame,
    team_id: str,
    mode: str,
    selected_match_ids: Optional[list[str]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    diagnostics = {
        "input_rows": int(len(touches_df)),
        "candidate_rows": 0,
        "candidate_rallies": 0,
        "phase_rallies": 0,
        "rallies_with_first_attack": 0,
        "counted_rallies": 0,
        "skipped_no_serve_row": 0,
        "skipped_no_first_attack": 0,
        "skipped_invalid_rotation": 0,
        "skipped_missing_point_winner": 0,
    }
    if touches_df.empty:
        return pd.DataFrame(), diagnostics

    team_id = str(team_id)
    df = touches_df.copy()
    df["home_team_id"] = df["home_team_id"].astype(str)
    df["visiting_team_id"] = df["visiting_team_id"].astype(str)

    if selected_match_ids:
        selected_set = {str(m) for m in selected_match_ids}
        df = df[df["match_alternative_id"].astype(str).isin(selected_set)]

    in_match_mask = (df["home_team_id"] == team_id) | (df["visiting_team_id"] == team_id)
    df = df.loc[in_match_mask].copy()
    diagnostics["candidate_rows"] = int(len(df))
    if df.empty:
        return pd.DataFrame(), diagnostics

    df["set_number"] = pd.to_numeric(df["set_number"], errors="coerce")
    df["rally_number"] = pd.to_numeric(df["rally_number"], errors="coerce")
    df = df[df["set_number"].notna() & df["rally_number"].notna()]
    df = df[(df["set_number"] > 0) & (df["rally_number"] > 0)]
    if df.empty:
        return pd.DataFrame(), diagnostics

    order_cols = [c for c in ["unique_row_id", "video_time"] if c in df.columns]
    if order_cols:
        df = df.sort_values(order_cols, kind="mergesort")
    else:
        df = df.sort_index()

    group_cols = ["match_alternative_id", "set_number", "rally_number"]
    diagnostics["candidate_rallies"] = int(df[group_cols].drop_duplicates().shape[0])
    rows: list[dict[str, object]] = []

    for (match_id, set_number, rally_number), g in df.groupby(group_cols, dropna=False, sort=False):
        first = g.iloc[0]
        is_home_selected = str(first.get("home_team_id")) == team_id
        selected_team_name = first.get("home_team") if is_home_selected else first.get("visiting_team")
        opponent_team_name = first.get("visiting_team") if is_home_selected else first.get("home_team")

        serving_values = g["serving_team"].dropna().astype(str)
        if serving_values.empty:
            continue
        serving_team = serving_values.iloc[0]

        if mode == "sideout":
            if serving_team == str(selected_team_name):
                continue
            receiving_team = selected_team_name
            phase = "selected_team_receiving"
        else:
            if serving_team != str(selected_team_name):
                continue
            receiving_team = opponent_team_name
            phase = "selected_team_serving"
        diagnostics["phase_rallies"] += 1

        setter_col = "home_setter_position" if is_home_selected else "visiting_setter_position"
        selected_rotation = _first_int(g[setter_col]) if setter_col in g.columns else None

        g_reset = g.reset_index(drop=True)
        serve_rows = g_reset[(g_reset["skill"] == "Serve") & (g_reset["team"] == serving_team)]
        if serve_rows.empty:
            diagnostics["skipped_no_serve_row"] += 1
            continue
        serve_pos = int(serve_rows.iloc[0].name)
        after_serve = g_reset.iloc[serve_pos + 1 :]

        receiving_attacks = after_serve[
            (after_serve["skill"] == "Attack")
            & (after_serve["team"] == receiving_team)
        ]
        if receiving_attacks.empty:
            diagnostics["skipped_no_first_attack"] += 1
            continue
        first_attack = receiving_attacks.iloc[0]
        diagnostics["rallies_with_first_attack"] += 1

        point_winner = _last_non_null(g_reset["point_won_by"])
        if point_winner is None:
            diagnostics["skipped_missing_point_winner"] += 1
            continue

        selected_team_point_won = int(point_winner == selected_team_name)

        attack_quality_raw = _normalize_eval(first_attack.get("evaluation_code"))
        attack_quality = (
            attack_quality_raw if attack_quality_raw in ATTACK_QUALITY_CODES else "OTHER"
        )
        first_attack_player = first_attack.get("player_name")

        rows.append(
            {
                "match_alternative_id": match_id,
                "set_number": int(set_number),
                "rally_number": int(rally_number),
                "rally_id": f"{match_id}|{int(set_number)}|{int(rally_number)}",
                "selected_team_id": team_id,
                "selected_team_name": selected_team_name,
                "mode": mode,
                "phase": phase,
                # In breakpoint mode this is explicitly the selected server rotation.
                "selected_rotation": selected_rotation,
                "first_attack_quality_raw": attack_quality_raw,
                "first_attack_quality": attack_quality,
                "first_attack_player_name": (
                    first_attack_player if pd.notna(first_attack_player) else "Unknown"
                ),
                "selected_team_point_won": selected_team_point_won,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out, diagnostics

    out["selected_rotation"] = pd.to_numeric(out["selected_rotation"], errors="coerce")
    before_rotation_filter = int(len(out))
    out = out[out["selected_rotation"].between(1, 6, inclusive="both")]
    diagnostics["skipped_invalid_rotation"] = before_rotation_filter - int(len(out))

    out["selected_rotation"] = out["selected_rotation"].astype(int)
    out["selected_team_point_won"] = (
        pd.to_numeric(out["selected_team_point_won"], errors="coerce").fillna(0).astype(int)
    )
    out = out.drop_duplicates(subset=["rally_id"], keep="first").reset_index(drop=True)
    diagnostics["counted_rallies"] = int(len(out))
    return out, diagnostics


def _build_quality_summary(rally_df: pd.DataFrame) -> pd.DataFrame:
    base = (
        rally_df.groupby("first_attack_quality", dropna=False)
        .agg(
            Attempts=("rally_id", "count"),
            Point_won_count=("selected_team_point_won", "sum"),
        )
        .reset_index()
        .rename(columns={"first_attack_quality": "Attack_quality"})
    )
    total_attempts = base["Attempts"].sum()
    base["Condition_share_of_first_attacks"] = np.where(
        total_attempts > 0,
        base["Attempts"] / total_attempts,
        np.nan,
    )
    base["Point_won_probability"] = np.where(
        base["Attempts"] > 0,
        base["Point_won_count"] / base["Attempts"],
        np.nan,
    )
    base = _sort_quality_column(base, "Attack_quality")

    total = pd.DataFrame(
        [
            {
                "Attack_quality": "Total",
                "Attempts": int(base["Attempts"].sum()),
                "Point_won_count": int(base["Point_won_count"].sum()),
                "Condition_share_of_first_attacks": 1.0 if total_attempts > 0 else np.nan,
                "Point_won_probability": (
                    base["Point_won_count"].sum() / base["Attempts"].sum()
                    if base["Attempts"].sum() > 0
                    else np.nan
                ),
            }
        ]
    )
    return pd.concat([base, total], ignore_index=True)


def _build_rotation_quality_summary(rally_df: pd.DataFrame) -> pd.DataFrame:
    base = (
        rally_df.groupby(["selected_rotation", "first_attack_quality"], dropna=False)
        .agg(
            Attempts=("rally_id", "count"),
            Point_won_count=("selected_team_point_won", "sum"),
        )
        .reset_index()
        .rename(columns={"selected_rotation": "Rotation", "first_attack_quality": "Attack_quality"})
    )
    total_attempts = base["Attempts"].sum()
    base["Condition_share_of_first_attacks"] = np.where(
        total_attempts > 0,
        base["Attempts"] / total_attempts,
        np.nan,
    )
    base["Condition_share_within_rotation"] = np.where(
        base.groupby("Rotation")["Attempts"].transform("sum") > 0,
        base["Attempts"] / base.groupby("Rotation")["Attempts"].transform("sum"),
        np.nan,
    )
    base["Point_won_probability"] = np.where(
        base["Attempts"] > 0,
        base["Point_won_count"] / base["Attempts"],
        np.nan,
    )
    base = _sort_quality_column(base, "Attack_quality")
    return base.sort_values(["Rotation", "Attack_quality"]).reset_index(drop=True)


def _build_rotation_probability_pivot(rotation_quality_summary: pd.DataFrame) -> pd.DataFrame:
    if rotation_quality_summary.empty:
        return pd.DataFrame()
    quality_order = ATTACK_QUALITY_CODES + ["OTHER"]
    pivot = (
        rotation_quality_summary.pivot(
            index="Rotation",
            columns="Attack_quality",
            values="Point_won_probability",
        )
        .reindex(index=range(1, 7), columns=quality_order)
    )
    pivot.index.name = "Rotation"
    return pivot


def _build_player_tables(rally_df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode != "sideout":
        return pd.DataFrame(), pd.DataFrame()

    base = rally_df.copy()
    if base.empty:
        return pd.DataFrame(), pd.DataFrame()

    by_player = (
        base.groupby("first_attack_player_name", dropna=False)
        .agg(
            Attempts=("rally_id", "count"),
            Point_won_count=("selected_team_point_won", "sum"),
        )
        .reset_index()
        .rename(columns={"first_attack_player_name": "Player"})
    )
    total_attempts = by_player["Attempts"].sum()
    by_player["Condition_share_of_first_attacks"] = np.where(
        total_attempts > 0,
        by_player["Attempts"] / total_attempts,
        np.nan,
    )
    by_player["Point_won_probability"] = np.where(
        by_player["Attempts"] > 0,
        by_player["Point_won_count"] / by_player["Attempts"],
        np.nan,
    )
    by_player = by_player.sort_values(["Attempts", "Point_won_count"], ascending=[False, False]).reset_index(drop=True)

    by_player_quality = (
        base.groupby(["first_attack_player_name", "first_attack_quality"], dropna=False)
        .agg(
            Attempts=("rally_id", "count"),
            Point_won_count=("selected_team_point_won", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "first_attack_player_name": "Player",
                "first_attack_quality": "Attack_quality",
            }
        )
    )
    total_attempts_by_player_quality = by_player_quality["Attempts"].sum()
    by_player_quality["Condition_share_of_first_attacks"] = np.where(
        total_attempts_by_player_quality > 0,
        by_player_quality["Attempts"] / total_attempts_by_player_quality,
        np.nan,
    )
    by_player_quality["Condition_share_within_player"] = np.where(
        by_player_quality.groupby("Player")["Attempts"].transform("sum") > 0,
        by_player_quality["Attempts"] / by_player_quality.groupby("Player")["Attempts"].transform("sum"),
        np.nan,
    )
    by_player_quality["Point_won_probability"] = np.where(
        by_player_quality["Attempts"] > 0,
        by_player_quality["Point_won_count"] / by_player_quality["Attempts"],
        np.nan,
    )
    by_player_quality = _sort_quality_column(by_player_quality, "Attack_quality")
    by_player_quality = by_player_quality.sort_values(["Player", "Attack_quality"]).reset_index(drop=True)

    return by_player, by_player_quality


def _sort_quality_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    order = ATTACK_QUALITY_CODES + ["OTHER"]
    out = df.copy()
    out["_q_rank"] = out[col].apply(lambda x: order.index(x) if x in order else len(order))
    out = out.sort_values("_q_rank").drop(columns="_q_rank")
    return out.reset_index(drop=True)


def _normalize_eval(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    return s[0]


def _last_non_null(series: pd.Series) -> Optional[object]:
    if series.empty:
        return None
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[-1]


def _first_int(series: pd.Series) -> Optional[int]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return None
    return int(vals.iloc[0])
