from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from dvolley.domain.breakpoint_touch_analysis import extract_team_matches


RECEPTION_CODES = ["#", "+", "!", "-", "/"]
BASE_CLASS_ORDER = [f"OUR RECEPTION {c}" for c in RECEPTION_CODES] + [
    "OPPONENT ACE",
    "OPPONENT SERVE ERROR",
]

ATTACK_ERROR_CODES = {"/", "="}


@dataclass
class SideoutTouchResult:
    receive_rallies: pd.DataFrame
    match_options: pd.DataFrame
    matrix_points_and_actions: pd.DataFrame
    matrix_code_counts: pd.DataFrame
    rotation_summary: pd.DataFrame
    attacker_summary: pd.DataFrame
    class_summary: pd.DataFrame
    class_rotation_tables: Dict[str, pd.DataFrame]
    class_order: list[str]
    diagnostics: dict[str, int]


def build_sideout_touch_analysis(
    touches_df: pd.DataFrame,
    team_id: str,
    selected_match_ids: Optional[Iterable[str]] = None,
) -> SideoutTouchResult:
    receive_rallies, diagnostics = _build_receive_rallies_df(touches_df, team_id, selected_match_ids)
    matches = extract_team_matches(touches_df, team_id)

    if receive_rallies.empty:
        empty = pd.DataFrame()
        return SideoutTouchResult(
            receive_rallies=receive_rallies,
            match_options=matches,
            matrix_points_and_actions=empty,
            matrix_code_counts=empty,
            rotation_summary=empty,
            attacker_summary=empty,
            class_summary=empty,
            class_rotation_tables={},
            class_order=BASE_CLASS_ORDER.copy(),
            diagnostics=diagnostics,
        )

    extra_classes = [
        c for c in sorted(receive_rallies["class_label"].dropna().unique()) if c not in BASE_CLASS_ORDER
    ]
    class_order = BASE_CLASS_ORDER + extra_classes

    by_rotation_class = (
        receive_rallies.groupby(["setter_position", "class_label"], dropna=False)
        .agg(
            Actions=("rally_id", "count"),
            Sideout_points=("sideout_point_for_class", "sum"),
        )
        .reset_index()
    )
    matrix_points_and_actions = _build_points_actions_matrix(by_rotation_class, class_order)
    matrix_code_counts = _build_code_count_matrix(by_rotation_class, class_order)
    rotation_summary = _build_rotation_summary(receive_rallies)
    attacker_summary = _build_attacker_summary(receive_rallies)
    class_summary = _build_class_summary(receive_rallies, class_order)
    class_rotation_tables = _build_class_rotation_tables(by_rotation_class, class_order)

    return SideoutTouchResult(
        receive_rallies=receive_rallies,
        match_options=matches,
        matrix_points_and_actions=matrix_points_and_actions,
        matrix_code_counts=matrix_code_counts,
        rotation_summary=rotation_summary,
        attacker_summary=attacker_summary,
        class_summary=class_summary,
        class_rotation_tables=class_rotation_tables,
        class_order=class_order,
        diagnostics=diagnostics,
    )


def _build_receive_rallies_df(
    touches_df: pd.DataFrame,
    team_id: str,
    selected_match_ids: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    diagnostics = {
        "input_rows": int(len(touches_df)),
        "candidate_rows": 0,
        "candidate_rallies": 0,
        "receiving_rallies": 0,
        "counted_rallies": 0,
        "skipped_no_serve_row": 0,
        "skipped_invalid_rotation": 0,
        "opponent_aces_from_serve_hash": 0,
        "opponent_aces_no_reception": 0,
        "opponent_serve_errors": 0,
        "missing_reception_rows": 0,
        "missing_first_attack_rows": 0,
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
    rows = []
    diagnostics["candidate_rallies"] = int(df[group_cols].drop_duplicates().shape[0])

    for (match_id, set_number, rally_number), g in df.groupby(group_cols, dropna=False, sort=False):
        first = g.iloc[0]
        is_home = str(first.get("home_team_id")) == team_id
        team_name = first.get("home_team") if is_home else first.get("visiting_team")
        serving_values = g["serving_team"].dropna().astype(str)
        if serving_values.empty:
            continue
        serving_team = serving_values.iloc[0]
        if serving_team == str(team_name):
            continue
        diagnostics["receiving_rallies"] += 1

        setter_col = "home_setter_position" if is_home else "visiting_setter_position"
        setter_pos = _first_int(g[setter_col]) if setter_col in g.columns else None

        g_reset = g.reset_index(drop=True)
        opp_serve_rows = g_reset[(g_reset["skill"] == "Serve") & (g_reset["team"] != team_name)]
        if opp_serve_rows.empty:
            diagnostics["skipped_no_serve_row"] += 1
            continue
        serve_row = opp_serve_rows.iloc[0]
        serve_pos = int(serve_row.name)
        serve_eval_code = _normalize_eval(serve_row.get("evaluation_code"))
        after_serve = g_reset.iloc[serve_pos + 1 :]

        point_winner = _last_non_null(g_reset["point_won_by"])
        sideout_point = int(point_winner == team_name) if point_winner is not None else 0

        our_receptions = after_serve[(after_serve["skill"] == "Reception") & (after_serve["team"] == team_name)]
        first_reception = our_receptions.iloc[0] if not our_receptions.empty else None
        reception_code = (
            _normalize_eval(first_reception.get("evaluation_code"))
            if first_reception is not None
            else None
        )

        our_real_actions = after_serve[
            (after_serve["team"] == team_name)
            & after_serve["skill"].notna()
            & (~after_serve["skill"].isin(["Point"]))
        ]
        no_reception_and_immediate_end = first_reception is None and our_real_actions.empty
        opponent_serve_error = int(serve_eval_code == "=")
        if opponent_serve_error:
            diagnostics["opponent_serve_errors"] += 1

        ace_from_serve_hash = sideout_point == 0 and serve_eval_code == "#"
        ace_no_reception = sideout_point == 0 and no_reception_and_immediate_end
        opponent_ace = int(ace_from_serve_hash or ace_no_reception)
        if ace_from_serve_hash:
            diagnostics["opponent_aces_from_serve_hash"] += 1
        if ace_no_reception:
            diagnostics["opponent_aces_no_reception"] += 1

        if opponent_serve_error:
            class_label = "OPPONENT SERVE ERROR"
        elif opponent_ace:
            class_label = "OPPONENT ACE"
        elif reception_code in RECEPTION_CODES:
            class_label = f"OUR RECEPTION {reception_code}"
        else:
            class_label = "OUR RECEPTION OTHER"
            diagnostics["missing_reception_rows"] += 1

        first_attack = None
        if first_reception is not None:
            reception_pos = int(first_reception.name)
            after_reception = g_reset.iloc[reception_pos + 1 :]
            our_attacks = after_reception[(after_reception["skill"] == "Attack") & (after_reception["team"] == team_name)]
            if not our_attacks.empty:
                first_attack = our_attacks.iloc[0]

        first_attack_player = None
        first_attack_eval = None
        first_attack_point = 0
        first_attack_block_or_error = 0
        if first_attack is not None:
            first_attack_player = first_attack.get("player_name")
            first_attack_eval = _normalize_eval(first_attack.get("evaluation_code"))
            first_attack_point = int(first_attack_eval == "#")
            first_attack_block_or_error = int(first_attack_eval in ATTACK_ERROR_CODES)
        elif first_reception is not None:
            diagnostics["missing_first_attack_rows"] += 1

        rows.append(
            {
                "match_alternative_id": match_id,
                "match_date": first.get("match_date"),
                "home_team": first.get("home_team"),
                "visiting_team": first.get("visiting_team"),
                "team_id": team_id,
                "team_name": team_name,
                "set_number": set_number,
                "rally_number": rally_number,
                "rally_id": f"{match_id}|{set_number}|{rally_number}",
                "setter_position": setter_pos,
                "serve_eval_code": serve_eval_code,
                "reception_code": reception_code,
                "class_label": class_label,
                "sideout_point": sideout_point,
                "opponent_serve_error": opponent_serve_error,
                "opponent_ace": opponent_ace,
                # Keep class-point aligned to your sheet: opponent serve errors have 0 points in class tables.
                "sideout_point_for_class": 0 if opponent_serve_error else sideout_point,
                "first_attack_player_name": (
                    first_attack_player if pd.notna(first_attack_player) else "Unknown"
                )
                if first_attack is not None
                else None,
                "first_attack_eval_code": first_attack_eval,
                "first_attack_point": first_attack_point,
                "first_attack_block_or_error": first_attack_block_or_error,
                "has_first_attack": int(first_attack is not None),
            }
        )
        diagnostics["counted_rallies"] += 1

    out = pd.DataFrame(rows)
    if out.empty:
        return out, diagnostics

    out["setter_position"] = pd.to_numeric(out["setter_position"], errors="coerce")
    before_rotation_filter = int(len(out))
    out = out[out["setter_position"].between(1, 6, inclusive="both")]
    diagnostics["skipped_invalid_rotation"] = before_rotation_filter - int(len(out))

    out["setter_position"] = out["setter_position"].astype(int)
    int_cols = [
        "sideout_point",
        "opponent_serve_error",
        "opponent_ace",
        "sideout_point_for_class",
        "first_attack_point",
        "first_attack_block_or_error",
        "has_first_attack",
    ]
    for col in int_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    out = out.drop_duplicates(subset=["rally_id"], keep="first").reset_index(drop=True)
    diagnostics["counted_rallies"] = int(len(out))
    return out, diagnostics


def _build_points_actions_matrix(by_rotation_class: pd.DataFrame, class_order: list[str]) -> pd.DataFrame:
    idx = range(1, 7)
    classes = class_order

    actions = (
        by_rotation_class.pivot(index="setter_position", columns="class_label", values="Actions")
        .reindex(index=idx, columns=classes)
        .fillna(0)
        .astype(int)
    )
    points = (
        by_rotation_class.pivot(index="setter_position", columns="class_label", values="Sideout_points")
        .reindex(index=idx, columns=classes)
        .fillna(0)
        .astype(int)
    )

    data = {}
    for c in classes:
        data[(c, "Sideout points")] = points[c]
        data[(c, "Actions")] = actions[c]
    out = pd.DataFrame(data, index=idx)
    out.index.name = "Rotation"

    total = pd.DataFrame([out.sum(numeric_only=True)], index=["Grand total"])
    return pd.concat([out, total], axis=0)


def _build_code_count_matrix(by_rotation_class: pd.DataFrame, class_order: list[str]) -> pd.DataFrame:
    idx = range(1, 7)
    count_pivot = (
        by_rotation_class.pivot(index="setter_position", columns="class_label", values="Actions")
        .reindex(index=idx, columns=class_order)
        .fillna(0)
        .astype(int)
    )
    count_pivot["Grand total"] = count_pivot.sum(axis=1)
    total_row = pd.DataFrame([count_pivot.sum()], index=["Grand total"])
    return pd.concat([count_pivot, total_row], axis=0)


def _build_rotation_summary(receive_rallies: pd.DataFrame) -> pd.DataFrame:
    idx = range(1, 7)
    action_mask = receive_rallies["class_label"] != "OPPONENT SERVE ERROR"

    grouped_actions = (
        receive_rallies[action_mask]
        .groupby("setter_position", dropna=False)
        .agg(
            Actions=("rally_id", "count"),
            Points=("sideout_point", "sum"),
        )
    )
    grouped_errors = (
        receive_rallies[~action_mask]
        .groupby("setter_position", dropna=False)
        .agg(
            Opponent_serve_errors=("rally_id", "count"),
        )
    )

    out = grouped_actions.join(grouped_errors, how="outer").reindex(idx).fillna(0).astype(int)
    out["% points WITH opp serve errors"] = np.where(
        (out["Actions"] + out["Opponent_serve_errors"]) > 0,
        (out["Points"] + out["Opponent_serve_errors"]) / (out["Actions"] + out["Opponent_serve_errors"]),
        np.nan,
    )
    out["% points WITHOUT opp serve errors"] = np.where(
        out["Actions"] > 0,
        out["Points"] / out["Actions"],
        np.nan,
    )
    out.index = [f"P{i}" for i in idx]

    total_actions = int(out["Actions"].sum())
    total_points = int(out["Points"].sum())
    total_opp_errors = int(out["Opponent_serve_errors"].sum())
    total_row = pd.DataFrame(
        [
            {
                "Actions": total_actions,
                "Points": total_points,
                "Opponent_serve_errors": total_opp_errors,
                "% points WITH opp serve errors": (
                    (total_points + total_opp_errors) / (total_actions + total_opp_errors)
                    if (total_actions + total_opp_errors) > 0
                    else np.nan
                ),
                "% points WITHOUT opp serve errors": (
                    total_points / total_actions if total_actions > 0 else np.nan
                ),
            }
        ],
        index=["Total"],
    )
    return pd.concat([out, total_row], axis=0)


def _build_attacker_summary(receive_rallies: pd.DataFrame) -> pd.DataFrame:
    attack_rows = receive_rallies[receive_rallies["has_first_attack"] == 1].copy()
    if attack_rows.empty:
        return pd.DataFrame(
            columns=[
                "player_name",
                "Attacks",
                "Points",
                "Blocked_or_errors",
                "% points on attacks",
                "% efficiency",
            ]
        )

    out = (
        attack_rows.groupby("first_attack_player_name", dropna=False)
        .agg(
            Attacks=("rally_id", "count"),
            Points=("first_attack_point", "sum"),
            Blocked_or_errors=("first_attack_block_or_error", "sum"),
        )
        .reset_index()
        .rename(columns={"first_attack_player_name": "player_name"})
    )
    out["% points on attacks"] = np.where(out["Attacks"] > 0, out["Points"] / out["Attacks"], np.nan)
    out["% efficiency"] = np.where(
        out["Attacks"] > 0,
        (out["Points"] - out["Blocked_or_errors"]) / out["Attacks"],
        np.nan,
    )
    return out.sort_values(["Attacks", "Points"], ascending=[False, False]).reset_index(drop=True)


def _build_class_summary(receive_rallies: pd.DataFrame, class_order: list[str]) -> pd.DataFrame:
    totals = (
        receive_rallies.groupby("class_label", dropna=False)
        .agg(
            Total_actions=("rally_id", "count"),
            Points_scored=("sideout_point_for_class", "sum"),
        )
        .reindex(class_order)
        .fillna(0)
    )
    totals["Total_actions"] = totals["Total_actions"].astype(int)
    totals["Points_scored"] = totals["Points_scored"].astype(int)

    total_actions = totals["Total_actions"].sum()
    total_points = totals["Points_scored"].sum()

    totals["% of total actions"] = np.where(
        total_actions > 0, totals["Total_actions"] / total_actions, np.nan
    )
    totals["% of total points"] = np.where(
        total_points > 0, totals["Points_scored"] / total_points, np.nan
    )
    totals["% points on actions"] = np.where(
        totals["Total_actions"] > 0,
        totals["Points_scored"] / totals["Total_actions"],
        np.nan,
    )

    total_row = pd.DataFrame(
        [
            {
                "Total_actions": int(total_actions),
                "Points_scored": int(total_points),
                "% of total actions": 1.0 if total_actions > 0 else np.nan,
                "% of total points": 1.0 if total_points > 0 else np.nan,
                "% points on actions": (total_points / total_actions) if total_actions > 0 else np.nan,
            }
        ],
        index=["Total"],
    )
    out = pd.concat([totals, total_row], axis=0)
    out.index.name = "Class"
    return out


def _build_class_rotation_tables(
    by_rotation_class: pd.DataFrame,
    class_order: list[str],
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    idx = range(1, 7)
    for class_name in class_order:
        class_df = by_rotation_class[by_rotation_class["class_label"] == class_name]
        if class_df.empty:
            continue
        base = class_df.set_index("setter_position")[["Sideout_points", "Actions"]].reindex(idx).fillna(0)
        base["Sideout_points"] = base["Sideout_points"].astype(int)
        base["Actions"] = base["Actions"].astype(int)
        base["%"] = np.where(base["Actions"] > 0, base["Sideout_points"] / base["Actions"], np.nan)
        base.index = [f"P{i}" for i in idx]

        total_row = pd.DataFrame(
            [
                {
                    "Sideout_points": int(base["Sideout_points"].sum()),
                    "Actions": int(base["Actions"].sum()),
                    "%": (
                        base["Sideout_points"].sum() / base["Actions"].sum()
                        if base["Actions"].sum() > 0
                        else np.nan
                    ),
                }
            ],
            index=["Total"],
        )
        out[class_name] = pd.concat([base, total_row], axis=0)
    return out


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
