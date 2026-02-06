from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


RECEPTION_CODES = ["#", "+", "!", "-", "/"]
BASE_CLASS_ORDER = [f"OPP RECEPTION {c}" for c in RECEPTION_CODES] + ["ACE", "ERRORS"]


@dataclass
class BreakpointTouchResult:
    serve_rallies: pd.DataFrame
    match_options: pd.DataFrame
    matrix_points_and_receptions: pd.DataFrame
    matrix_code_counts: pd.DataFrame
    rotation_summary: pd.DataFrame
    player_summary: pd.DataFrame
    class_summary: pd.DataFrame
    class_rotation_tables: Dict[str, pd.DataFrame]
    class_order: list[str]
    diagnostics: dict[str, int]


def extract_team_catalog(touches_df: pd.DataFrame) -> pd.DataFrame:
    if touches_df.empty:
        return pd.DataFrame(columns=["team_id", "team_name"])

    home = (
        touches_df[["home_team_id", "home_team"]]
        .dropna(subset=["home_team_id"])
        .rename(columns={"home_team_id": "team_id", "home_team": "team_name"})
    )
    away = (
        touches_df[["visiting_team_id", "visiting_team"]]
        .dropna(subset=["visiting_team_id"])
        .rename(columns={"visiting_team_id": "team_id", "visiting_team": "team_name"})
    )
    teams = pd.concat([home, away], ignore_index=True)
    teams["team_id"] = teams["team_id"].astype(str)
    teams["team_name"] = teams["team_name"].fillna("Unknown").astype(str)
    teams = teams.drop_duplicates(["team_id", "team_name"])
    return teams.sort_values(["team_name", "team_id"]).reset_index(drop=True)


def extract_team_matches(touches_df: pd.DataFrame, team_id: str) -> pd.DataFrame:
    if touches_df.empty:
        return pd.DataFrame(
            columns=["match_alternative_id", "match_date", "home_team", "visiting_team", "label"]
        )

    team_id = str(team_id)
    df = touches_df.copy()
    df["home_team_id"] = df["home_team_id"].astype(str)
    df["visiting_team_id"] = df["visiting_team_id"].astype(str)

    match_mask = (df["home_team_id"] == team_id) | (df["visiting_team_id"] == team_id)
    subset = df.loc[match_mask].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=["match_alternative_id", "match_date", "home_team", "visiting_team", "label"]
        )

    base = (
        subset.groupby("match_alternative_id", dropna=False)
        .agg(
            match_date=("match_date", "first"),
            home_team=("home_team", "first"),
            visiting_team=("visiting_team", "first"),
        )
        .reset_index()
    )
    base["match_date"] = base["match_date"].fillna("Unknown").astype(str)
    base["label"] = (
        base["match_date"].astype(str)
        + " | "
        + base["home_team"].fillna("Unknown").astype(str)
        + " vs "
        + base["visiting_team"].fillna("Unknown").astype(str)
    )
    return base.sort_values("match_date").reset_index(drop=True)


def build_breakpoint_touch_analysis(
    touches_df: pd.DataFrame,
    team_id: str,
    selected_match_ids: Optional[Iterable[str]] = None,
) -> BreakpointTouchResult:
    serve_rallies, diagnostics = _build_serve_rallies_df(touches_df, team_id, selected_match_ids)
    matches = extract_team_matches(touches_df, team_id)

    if serve_rallies.empty:
        empty = pd.DataFrame()
        return BreakpointTouchResult(
            serve_rallies=serve_rallies,
            match_options=matches,
            matrix_points_and_receptions=empty,
            matrix_code_counts=empty,
            rotation_summary=empty,
            player_summary=empty,
            class_summary=empty,
            class_rotation_tables={},
            class_order=BASE_CLASS_ORDER.copy(),
            diagnostics=diagnostics,
        )

    extra_classes = [
        c for c in sorted(serve_rallies["class_label"].dropna().unique()) if c not in BASE_CLASS_ORDER
    ]
    class_order = BASE_CLASS_ORDER + extra_classes

    by_rotation_class = (
        serve_rallies.groupby(["setter_position", "class_label"], dropna=False)
        .agg(
            Receptions=("rally_id", "count"),
            Our_points=("break_point", "sum"),
        )
        .reset_index()
    )
    matrix_points_and_receptions = _build_points_receptions_matrix(by_rotation_class, class_order)
    matrix_code_counts = _build_code_count_matrix(by_rotation_class, class_order)
    rotation_summary = _build_rotation_summary(serve_rallies)
    player_summary = _build_player_summary(serve_rallies)
    class_summary = _build_class_summary(serve_rallies, class_order)
    class_rotation_tables = _build_class_rotation_tables(by_rotation_class, class_order)

    return BreakpointTouchResult(
        serve_rallies=serve_rallies,
        match_options=matches,
        matrix_points_and_receptions=matrix_points_and_receptions,
        matrix_code_counts=matrix_code_counts,
        rotation_summary=rotation_summary,
        player_summary=player_summary,
        class_summary=class_summary,
        class_rotation_tables=class_rotation_tables,
        class_order=class_order,
        diagnostics=diagnostics,
    )


def _build_serve_rallies_df(
    touches_df: pd.DataFrame,
    team_id: str,
    selected_match_ids: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    diagnostics = {
        "input_rows": int(len(touches_df)),
        "candidate_rows": 0,
        "candidate_rallies": 0,
        "served_rallies": 0,
        "counted_rallies": 0,
        "skipped_no_serve_row": 0,
        "skipped_invalid_rotation": 0,
        "ace_from_serve_hash": 0,
        "ace_from_no_pass_immediate_end": 0,
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
        if serving_team != str(team_name):
            continue
        diagnostics["served_rallies"] += 1

        setter_col = "home_setter_position" if is_home else "visiting_setter_position"
        setter_pos = _first_int(g[setter_col]) if setter_col in g.columns else None

        serve_rows = g[(g["skill"] == "Serve") & (g["team"] == team_name)]
        if serve_rows.empty:
            diagnostics["skipped_no_serve_row"] += 1
            continue
        serve_row = serve_rows.iloc[0]
        serve_eval_code = _normalize_eval(serve_row.get("evaluation_code"))
        server_player = serve_row.get("player_name")

        point_winner = _last_non_null(g["point_won_by"])
        break_point = int(point_winner == team_name) if point_winner is not None else 0

        g_reset = g.reset_index(drop=True)
        serve_idx = []
        if "unique_row_id" in g_reset.columns and serve_row.get("unique_row_id") is not None:
            serve_idx = g_reset.index[g_reset["unique_row_id"] == serve_row.get("unique_row_id")].tolist()
        if not serve_idx:
            serve_idx = g_reset.index[
                (g_reset["skill"] == "Serve") & (g_reset["team"] == team_name)
            ].tolist()
        serve_pos = serve_idx[0] if serve_idx else 0
        after_serve = g_reset.iloc[serve_pos + 1 :]

        opp_receptions = after_serve[(after_serve["skill"] == "Reception") & (after_serve["team"] != team_name)]
        first_reception = opp_receptions.iloc[0] if not opp_receptions.empty else None
        reception_code = (
            _normalize_eval(first_reception.get("evaluation_code"))
            if first_reception is not None
            else None
        )

        opp_real_actions = after_serve[
            (after_serve["team"] != team_name)
            & after_serve["skill"].notna()
            & (~after_serve["skill"].isin(["Point"]))
        ]
        no_pass_and_immediate_end = first_reception is None and opp_real_actions.empty

        # ACE must come from serve outcome (or direct no-pass immediate point), not from reception '#'.
        ace_from_serve_hash = break_point == 1 and serve_eval_code == "#"
        ace_from_no_pass = break_point == 1 and no_pass_and_immediate_end
        is_ace = ace_from_serve_hash or ace_from_no_pass
        if ace_from_serve_hash:
            diagnostics["ace_from_serve_hash"] += 1
        if ace_from_no_pass:
            diagnostics["ace_from_no_pass_immediate_end"] += 1
        is_error = serve_eval_code == "=" or (
            break_point == 0 and first_reception is None and no_pass_and_immediate_end
        )

        if is_error:
            class_label = "ERRORS"
        elif is_ace:
            class_label = "ACE"
        elif reception_code in RECEPTION_CODES:
            class_label = f"OPP RECEPTION {reception_code}"
        else:
            class_label = "OPP RECEPTION OTHER"

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
                "break_point": break_point,
                "serve_error": int(is_error),
                "server_player_name": server_player if pd.notna(server_player) else "Unknown",
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
    out["break_point"] = out["break_point"].astype(int)
    out["serve_error"] = out["serve_error"].astype(int)
    out = out.drop_duplicates(subset=["rally_id"], keep="first").reset_index(drop=True)
    diagnostics["counted_rallies"] = int(len(out))
    return out, diagnostics


def _build_points_receptions_matrix(by_rotation_class: pd.DataFrame, class_order: list[str]) -> pd.DataFrame:
    idx = range(1, 7)
    classes = class_order

    receptions = (
        by_rotation_class.pivot(index="setter_position", columns="class_label", values="Receptions")
        .reindex(index=idx, columns=classes)
        .fillna(0)
        .astype(int)
    )
    points = (
        by_rotation_class.pivot(index="setter_position", columns="class_label", values="Our_points")
        .reindex(index=idx, columns=classes)
        .fillna(0)
        .astype(int)
    )

    data = {}
    for c in classes:
        data[(c, "Our points")] = points[c]
        data[(c, "Receptions")] = receptions[c]
    out = pd.DataFrame(data, index=idx)
    out.index.name = "Rotation"

    total = pd.DataFrame([out.sum(numeric_only=True)], index=["Grand total"])
    return pd.concat([out, total], axis=0)


def _build_code_count_matrix(by_rotation_class: pd.DataFrame, class_order: list[str]) -> pd.DataFrame:
    idx = range(1, 7)
    count_pivot = (
        by_rotation_class.pivot(index="setter_position", columns="class_label", values="Receptions")
        .reindex(index=idx, columns=class_order)
        .fillna(0)
        .astype(int)
    )
    count_pivot["Grand total"] = count_pivot.sum(axis=1)
    total_row = pd.DataFrame([count_pivot.sum()], index=["Grand total"])
    return pd.concat([count_pivot, total_row], axis=0)


def _build_rotation_summary(serve_rallies: pd.DataFrame) -> pd.DataFrame:
    idx = range(1, 7)
    grouped = serve_rallies.groupby("setter_position", dropna=False).agg(
        Serves=("rally_id", "count"),
        Break_points=("break_point", "sum"),
        Errors=("serve_error", "sum"),
    )
    grouped = grouped.reindex(idx).fillna(0).astype(int)
    out = grouped.copy()
    out["% break on serves"] = np.where(out["Serves"] > 0, out["Break_points"] / out["Serves"], np.nan)
    out["% errors"] = np.where(out["Serves"] > 0, out["Errors"] / out["Serves"], np.nan)
    out["% break excl errors"] = np.where(
        (out["Serves"] - out["Errors"]) > 0,
        out["Break_points"] / (out["Serves"] - out["Errors"]),
        np.nan,
    )
    out.index = [f"P{i}" for i in idx]

    total_serves = int(out["Serves"].sum())
    total_break = int(out["Break_points"].sum())
    total_errors = int(out["Errors"].sum())
    total_row = pd.DataFrame(
        [
            {
                "Serves": total_serves,
                "Break_points": total_break,
                "Errors": total_errors,
                "% break on serves": (total_break / total_serves) if total_serves else np.nan,
                "% errors": (total_errors / total_serves) if total_serves else np.nan,
                "% break excl errors": (
                    total_break / (total_serves - total_errors)
                    if (total_serves - total_errors) > 0
                    else np.nan
                ),
            }
        ],
        index=["Total"],
    )
    return pd.concat([out, total_row], axis=0)


def _build_player_summary(serve_rallies: pd.DataFrame) -> pd.DataFrame:
    if serve_rallies.empty:
        return pd.DataFrame(columns=["player_name", "Serves", "Break_points", "% break on serves"])

    out = (
        serve_rallies.groupby("server_player_name", dropna=False)
        .agg(
            Serves=("rally_id", "count"),
            Break_points=("break_point", "sum"),
        )
        .reset_index()
        .rename(columns={"server_player_name": "player_name"})
    )
    out["% break on serves"] = np.where(out["Serves"] > 0, out["Break_points"] / out["Serves"], np.nan)
    return out.sort_values(["Serves", "Break_points"], ascending=[False, False]).reset_index(drop=True)


def _build_class_summary(serve_rallies: pd.DataFrame, class_order: list[str]) -> pd.DataFrame:
    totals = (
        serve_rallies.groupby("class_label", dropna=False)
        .agg(
            Total_serves=("rally_id", "count"),
            Points_scored=("break_point", "sum"),
        )
        .reindex(class_order)
        .fillna(0)
    )
    totals["Total_serves"] = totals["Total_serves"].astype(int)
    totals["Points_scored"] = totals["Points_scored"].astype(int)

    total_serves = totals["Total_serves"].sum()
    total_points = totals["Points_scored"].sum()

    totals["% of total serves"] = np.where(total_serves > 0, totals["Total_serves"] / total_serves, np.nan)
    totals["% of total points"] = np.where(total_points > 0, totals["Points_scored"] / total_points, np.nan)
    totals["% points on serves"] = np.where(
        totals["Total_serves"] > 0,
        totals["Points_scored"] / totals["Total_serves"],
        np.nan,
    )

    total_row = pd.DataFrame(
        [
            {
                "Total_serves": int(total_serves),
                "Points_scored": int(total_points),
                "% of total serves": 1.0 if total_serves > 0 else np.nan,
                "% of total points": 1.0 if total_points > 0 else np.nan,
                "% points on serves": (total_points / total_serves) if total_serves > 0 else np.nan,
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
        base = class_df.set_index("setter_position")[["Our_points", "Receptions"]].reindex(idx).fillna(0)
        base["Our_points"] = base["Our_points"].astype(int)
        base["Receptions"] = base["Receptions"].astype(int)
        base["%"] = np.where(base["Receptions"] > 0, base["Our_points"] / base["Receptions"], np.nan)
        base.index = [f"P{i}" for i in idx]

        total_row = pd.DataFrame(
            [
                {
                    "Our_points": int(base["Our_points"].sum()),
                    "Receptions": int(base["Receptions"].sum()),
                    "%": (
                        base["Our_points"].sum() / base["Receptions"].sum()
                        if base["Receptions"].sum() > 0
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
