from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from dvolley.domain.bayesian_stats import add_beta_interval_columns


PASS_QUALITY_ORDER = ["#", "+", "!", "-", "/", "OTHER"]
ATTACK_QUALITY_ORDER = ["#", "+", "!", "-", "/", "=", "OTHER"]


@dataclass
class PlayerSideoutDataset:
    sideout_rallies: pd.DataFrame
    team_attacks: pd.DataFrame
    players: list[str]
    diagnostics: dict[str, int]


@dataclass
class PlayerAnalysisTables:
    first_attack_overall: pd.DataFrame
    first_attack_by_pass_quality: dict[str, pd.DataFrame]
    total_attack_table: pd.DataFrame
    non_first_attack_table: pd.DataFrame
    pass_quality_table: pd.DataFrame
    first_attack_attempts: int
    total_attack_attempts: int
    non_first_attack_attempts: int
    pass_attempts: int


def build_player_sideout_dataset(
    touches_df: pd.DataFrame,
    team_id: str,
    selected_match_ids: Optional[Iterable[str]] = None,
) -> PlayerSideoutDataset:
    diagnostics = {
        "input_rows": int(len(touches_df)),
        "candidate_rows": 0,
        "candidate_rallies": 0,
        "sideout_rallies": 0,
        "counted_rallies": 0,
        "attack_rows": 0,
        "first_pass_rows": 0,
        "first_attack_rows": 0,
        "skipped_no_serve_row": 0,
        "skipped_invalid_rotation": 0,
        "skipped_missing_point_winner": 0,
    }
    if touches_df.empty:
        return PlayerSideoutDataset(
            sideout_rallies=pd.DataFrame(),
            team_attacks=pd.DataFrame(),
            players=[],
            diagnostics=diagnostics,
        )

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
        return PlayerSideoutDataset(
            sideout_rallies=pd.DataFrame(),
            team_attacks=pd.DataFrame(),
            players=[],
            diagnostics=diagnostics,
        )

    df["set_number"] = pd.to_numeric(df["set_number"], errors="coerce")
    df["rally_number"] = pd.to_numeric(df["rally_number"], errors="coerce")
    df = df[df["set_number"].notna() & df["rally_number"].notna()]
    df = df[(df["set_number"] > 0) & (df["rally_number"] > 0)]
    if df.empty:
        return PlayerSideoutDataset(
            sideout_rallies=pd.DataFrame(),
            team_attacks=pd.DataFrame(),
            players=[],
            diagnostics=diagnostics,
        )

    order_cols = [c for c in ["unique_row_id", "video_time"] if c in df.columns]
    if order_cols:
        df = df.sort_values(order_cols, kind="mergesort")
    else:
        df = df.sort_index()

    group_cols = ["match_alternative_id", "set_number", "rally_number"]
    diagnostics["candidate_rallies"] = int(df[group_cols].drop_duplicates().shape[0])
    rally_rows: list[dict[str, object]] = []
    attack_rows: list[dict[str, object]] = []

    for (match_id, set_number, rally_number), g in df.groupby(group_cols, dropna=False, sort=False):
        first = g.iloc[0]
        is_home_selected = str(first.get("home_team_id")) == team_id
        selected_team_name = first.get("home_team") if is_home_selected else first.get("visiting_team")
        setter_col = "home_setter_position" if is_home_selected else "visiting_setter_position"
        selected_rotation = _first_int(g[setter_col]) if setter_col in g.columns else None

        serving_values = g["serving_team"].dropna().astype(str)
        if serving_values.empty:
            continue
        serving_team = serving_values.iloc[0]
        phase = "breakpoint" if serving_team == str(selected_team_name) else "sideout"

        g_reset = g.reset_index(drop=True)
        point_winner = _last_non_null(g_reset["point_won_by"])
        if point_winner is None:
            diagnostics["skipped_missing_point_winner"] += 1
            rally_won = 0
        else:
            rally_won = int(point_winner == selected_team_name)

        serve_rows = g_reset[(g_reset["skill"] == "Serve") & (g_reset["team"] == serving_team)]
        if serve_rows.empty:
            diagnostics["skipped_no_serve_row"] += 1
            serve_pos = 0
        else:
            serve_pos = int(serve_rows.iloc[0].name)
        after_serve = g_reset.iloc[serve_pos + 1 :]

        first_pass_quality = None
        first_attack_idx = None

        if phase == "sideout":
            diagnostics["sideout_rallies"] += 1
            reception_rows = after_serve[
                (after_serve["skill"] == "Reception") & (after_serve["team"] == selected_team_name)
            ]
            first_reception = reception_rows.iloc[0] if not reception_rows.empty else None
            first_pass_player = (
                _normalize_player_name(first_reception.get("player_name"))
                if first_reception is not None
                else None
            )
            first_pass_quality = (
                _normalize_eval_code(first_reception.get("evaluation_code"), PASS_QUALITY_ORDER)
                if first_reception is not None
                else None
            )
            if first_reception is not None:
                diagnostics["first_pass_rows"] += 1

            first_attack = None
            if first_reception is not None:
                reception_pos = int(first_reception.name)
                after_reception = g_reset.iloc[reception_pos + 1 :]
                attacks_after_reception = after_reception[
                    (after_reception["skill"] == "Attack") & (after_reception["team"] == selected_team_name)
                ]
                if not attacks_after_reception.empty:
                    first_attack = attacks_after_reception.iloc[0]
                    first_attack_idx = int(first_attack.name)

            first_attack_player = (
                _normalize_player_name(first_attack.get("player_name"))
                if first_attack is not None
                else None
            )
            first_attack_quality = (
                _normalize_eval_code(first_attack.get("evaluation_code"), ATTACK_QUALITY_ORDER)
                if first_attack is not None
                else None
            )
            if first_attack is not None:
                diagnostics["first_attack_rows"] += 1

            rally_rows.append(
                {
                    "rally_id": f"{match_id}|{int(set_number)}|{int(rally_number)}",
                    "match_alternative_id": match_id,
                    "set_number": int(set_number),
                    "rally_number": int(rally_number),
                    "rotation": selected_rotation,
                    "sideout_point": rally_won,
                    "first_pass_player": first_pass_player,
                    "first_pass_quality": first_pass_quality,
                    "first_attack_player": first_attack_player,
                    "first_attack_quality": first_attack_quality,
                }
            )

        team_attacks_after_serve = after_serve[
            (after_serve["skill"] == "Attack") & (after_serve["team"] == selected_team_name)
        ]
        for _, attack_row in team_attacks_after_serve.iterrows():
            attack_quality = _normalize_eval_code(attack_row.get("evaluation_code"), ATTACK_QUALITY_ORDER)
            attack_rows.append(
                {
                    "rally_id": f"{match_id}|{int(set_number)}|{int(rally_number)}",
                    "match_alternative_id": match_id,
                    "set_number": int(set_number),
                    "rally_number": int(rally_number),
                    "rotation": selected_rotation,
                    "phase": phase,
                    "rally_won": rally_won,
                    "player_name": _normalize_player_name(attack_row.get("player_name")),
                    "attack_quality": attack_quality,
                    "attack_point": int(attack_quality == "#"),
                    "is_first_attack": int(
                        phase == "sideout"
                        and first_attack_idx is not None
                        and int(attack_row.name) == int(first_attack_idx)
                    ),
                    "first_pass_quality": first_pass_quality,
                }
            )

    sideout_rallies = pd.DataFrame(rally_rows)
    team_attacks = pd.DataFrame(attack_rows)

    if sideout_rallies.empty:
        players = (
            sorted([str(v) for v in team_attacks["player_name"].dropna().unique().tolist()])
            if not team_attacks.empty
            else []
        )
        diagnostics["attack_rows"] = int(len(team_attacks))
        return PlayerSideoutDataset(
            sideout_rallies=sideout_rallies,
            team_attacks=team_attacks,
            players=players,
            diagnostics=diagnostics,
        )

    sideout_rallies["rotation"] = pd.to_numeric(sideout_rallies["rotation"], errors="coerce")
    before_rotation_filter = int(len(sideout_rallies))
    sideout_rallies = sideout_rallies[sideout_rallies["rotation"].between(1, 6, inclusive="both")]
    diagnostics["skipped_invalid_rotation"] = before_rotation_filter - int(len(sideout_rallies))
    sideout_rallies["rotation"] = sideout_rallies["rotation"].astype(int)
    sideout_rallies["sideout_point"] = (
        pd.to_numeric(sideout_rallies["sideout_point"], errors="coerce").fillna(0).astype(int)
    )
    sideout_rallies = sideout_rallies.drop_duplicates(subset=["rally_id"], keep="first").reset_index(drop=True)

    if not team_attacks.empty:
        team_attacks["rotation"] = pd.to_numeric(team_attacks["rotation"], errors="coerce")
        team_attacks = team_attacks[team_attacks["rotation"].between(1, 6, inclusive="both")]
        team_attacks["rotation"] = team_attacks["rotation"].astype(int)
        team_attacks["rally_won"] = (
            pd.to_numeric(team_attacks["rally_won"], errors="coerce").fillna(0).astype(int)
        )
        team_attacks["attack_point"] = (
            pd.to_numeric(team_attacks["attack_point"], errors="coerce").fillna(0).astype(int)
        )
        team_attacks["is_first_attack"] = (
            pd.to_numeric(team_attacks["is_first_attack"], errors="coerce").fillna(0).astype(int)
        )
        team_attacks = team_attacks.reset_index(drop=True)

    players = sorted(
        {
            *[str(v) for v in team_attacks["player_name"].dropna().unique().tolist()],
            *[str(v) for v in sideout_rallies["first_pass_player"].dropna().unique().tolist()],
            *[str(v) for v in sideout_rallies["first_attack_player"].dropna().unique().tolist()],
        }
    )

    diagnostics["counted_rallies"] = int(len(sideout_rallies))
    diagnostics["attack_rows"] = int(len(team_attacks))
    diagnostics["first_pass_rows"] = int(sideout_rallies["first_pass_player"].notna().sum())
    diagnostics["first_attack_rows"] = int(sideout_rallies["first_attack_player"].notna().sum())

    return PlayerSideoutDataset(
        sideout_rallies=sideout_rallies,
        team_attacks=team_attacks,
        players=players,
        diagnostics=diagnostics,
    )


def build_player_analysis_tables(
    dataset: PlayerSideoutDataset,
    player_name: str,
    *,
    include_by_rotation: bool = True,
) -> PlayerAnalysisTables:
    player_name = str(player_name)
    sideout_rallies = dataset.sideout_rallies.copy()
    team_attacks = dataset.team_attacks.copy()

    first_attack_rows = sideout_rallies[
        (sideout_rallies["first_attack_player"] == player_name)
        & sideout_rallies["first_attack_quality"].notna()
    ].copy()
    attack_quality_order = _ordered_codes(first_attack_rows["first_attack_quality"], ATTACK_QUALITY_ORDER)
    first_attack_overall = _build_condition_table(
        first_attack_rows,
        condition_col="first_attack_quality",
        condition_order=attack_quality_order,
        success_col=None,
        rally_win_col="sideout_point",
        include_by_rotation=include_by_rotation,
    )

    first_attack_by_pass_quality: dict[str, pd.DataFrame] = {}
    pass_order = _ordered_codes(first_attack_rows["first_pass_quality"], PASS_QUALITY_ORDER)
    for pass_quality in pass_order:
        subset = first_attack_rows[first_attack_rows["first_pass_quality"] == pass_quality].copy()
        table = _build_condition_table(
            subset,
            condition_col="first_attack_quality",
            condition_order=attack_quality_order,
            success_col=None,
            rally_win_col="sideout_point",
            include_by_rotation=include_by_rotation,
        )
        if not table.empty:
            first_attack_by_pass_quality[pass_quality] = table

    total_attack_rows = team_attacks[
        (team_attacks["player_name"] == player_name)
        & team_attacks["attack_quality"].notna()
    ].copy()
    total_attack_order = _ordered_codes(total_attack_rows["attack_quality"], ATTACK_QUALITY_ORDER)
    total_attack_table = _build_condition_table(
        total_attack_rows,
        condition_col="attack_quality",
        condition_order=total_attack_order,
        success_col=None,
        rally_win_col="rally_won",
        include_by_rotation=include_by_rotation,
    )

    non_first_attack_rows = team_attacks[
        (team_attacks["player_name"] == player_name)
        & (team_attacks["is_first_attack"] == 0)
        & team_attacks["attack_quality"].notna()
    ].copy()
    non_first_order = _ordered_codes(non_first_attack_rows["attack_quality"], ATTACK_QUALITY_ORDER)
    non_first_attack_table = _build_condition_table(
        non_first_attack_rows,
        condition_col="attack_quality",
        condition_order=non_first_order,
        success_col=None,
        rally_win_col="rally_won",
        include_by_rotation=include_by_rotation,
    )

    pass_rows = sideout_rallies[
        (sideout_rallies["first_pass_player"] == player_name)
        & sideout_rallies["first_pass_quality"].notna()
    ].copy()
    player_pass_order = _ordered_codes(pass_rows["first_pass_quality"], PASS_QUALITY_ORDER)
    pass_quality_table = _build_condition_table(
        pass_rows,
        condition_col="first_pass_quality",
        condition_order=player_pass_order,
        success_col=None,
        rally_win_col="sideout_point",
        include_by_rotation=include_by_rotation,
    )

    return PlayerAnalysisTables(
        first_attack_overall=first_attack_overall,
        first_attack_by_pass_quality=first_attack_by_pass_quality,
        total_attack_table=total_attack_table,
        non_first_attack_table=non_first_attack_table,
        pass_quality_table=pass_quality_table,
        first_attack_attempts=int(len(first_attack_rows)),
        total_attack_attempts=int(len(total_attack_rows)),
        non_first_attack_attempts=int(len(non_first_attack_rows)),
        pass_attempts=int(len(pass_rows)),
    )


def _build_condition_table(
    df: pd.DataFrame,
    condition_col: str,
    condition_order: list[str],
    *,
    success_col: str | None = None,
    rally_win_col: str | None = None,
    include_by_rotation: bool,
) -> pd.DataFrame:
    if df.empty or not condition_order:
        return pd.DataFrame()

    segments: list[tuple[str, int | None]] = [("Total", None)]
    if include_by_rotation:
        segments.extend((f"P{rot}", rot) for rot in range(1, 7))

    rows = []
    for condition_value in condition_order:
        row = {}
        for segment_name, rot in segments:
            segment_df = df if rot is None else df[df["rotation"] == rot]
            denominator = len(segment_df)
            condition_df = segment_df[segment_df[condition_col] == condition_value]
            actions = int(len(condition_df))
            share = actions / denominator if denominator > 0 else np.nan

            row[(segment_name, "Actions")] = actions
            row[(segment_name, "% share")] = share
            if success_col:
                success = int(condition_df[success_col].sum()) if actions else 0
                success_rate = success / actions if actions > 0 else np.nan
                row[(segment_name, "Successful")] = success
                row[(segment_name, "% successful")] = success_rate
            if rally_win_col:
                rallies_won = int(condition_df[rally_win_col].sum()) if actions else 0
                rallies_won_rate = rallies_won / actions if actions > 0 else np.nan
                row[(segment_name, "Rallies won")] = rallies_won
                row[(segment_name, "% rally won")] = rallies_won_rate

            share_stats = add_beta_interval_columns(
                pd.DataFrame(
                    [
                        {
                            "Actions": actions,
                            "Denominator": denominator if denominator > 0 else np.nan,
                        }
                    ]
                ),
                successes_col="Actions",
                trials_col="Denominator",
                prefix="% share",
            ).iloc[0]
            row[(segment_name, "% share 95% CI low")] = share_stats["% share 95% CI low"]
            row[(segment_name, "% share 95% CI high")] = share_stats["% share 95% CI high"]

            if success_col:
                success_stats = add_beta_interval_columns(
                    pd.DataFrame(
                        [
                            {
                                "Successful": success,
                                "Actions": actions if actions > 0 else np.nan,
                            }
                        ]
                    ),
                    successes_col="Successful",
                    trials_col="Actions",
                    prefix="% successful",
                ).iloc[0]
                row[(segment_name, "% successful 95% CI low")] = success_stats[
                    "% successful 95% CI low"
                ]
                row[(segment_name, "% successful 95% CI high")] = success_stats[
                    "% successful 95% CI high"
                ]
            if rally_win_col:
                rally_stats = add_beta_interval_columns(
                    pd.DataFrame(
                        [
                            {
                                "Rallies won": rallies_won,
                                "Actions": actions if actions > 0 else np.nan,
                            }
                        ]
                    ),
                    successes_col="Rallies won",
                    trials_col="Actions",
                    prefix="% rally won",
                ).iloc[0]
                row[(segment_name, "% rally won 95% CI low")] = rally_stats[
                    "% rally won 95% CI low"
                ]
                row[(segment_name, "% rally won 95% CI high")] = rally_stats[
                    "% rally won 95% CI high"
                ]
        rows.append(row)

    total_row = {}
    for segment_name, rot in segments:
        segment_df = df if rot is None else df[df["rotation"] == rot]
        denominator = len(segment_df)
        actions = int(denominator)
        share = actions / denominator if denominator > 0 else np.nan

        total_row[(segment_name, "Actions")] = actions
        total_row[(segment_name, "% share")] = share
        if success_col:
            success = int(segment_df[success_col].sum()) if actions else 0
            success_rate = success / actions if actions > 0 else np.nan
            total_row[(segment_name, "Successful")] = success
            total_row[(segment_name, "% successful")] = success_rate
        if rally_win_col:
            rallies_won = int(segment_df[rally_win_col].sum()) if actions else 0
            rallies_won_rate = rallies_won / actions if actions > 0 else np.nan
            total_row[(segment_name, "Rallies won")] = rallies_won
            total_row[(segment_name, "% rally won")] = rallies_won_rate

        share_stats = add_beta_interval_columns(
            pd.DataFrame(
                [
                    {
                        "Actions": actions,
                        "Denominator": denominator if denominator > 0 else np.nan,
                    }
                ]
            ),
            successes_col="Actions",
            trials_col="Denominator",
            prefix="% share",
        ).iloc[0]
        total_row[(segment_name, "% share 95% CI low")] = share_stats["% share 95% CI low"]
        total_row[(segment_name, "% share 95% CI high")] = share_stats["% share 95% CI high"]

        if success_col:
            success_stats = add_beta_interval_columns(
                pd.DataFrame(
                    [
                        {
                            "Successful": success,
                            "Actions": actions if actions > 0 else np.nan,
                        }
                    ]
                ),
                successes_col="Successful",
                trials_col="Actions",
                prefix="% successful",
            ).iloc[0]
            total_row[(segment_name, "% successful 95% CI low")] = success_stats[
                "% successful 95% CI low"
            ]
            total_row[(segment_name, "% successful 95% CI high")] = success_stats[
                "% successful 95% CI high"
            ]
        if rally_win_col:
            rally_stats = add_beta_interval_columns(
                pd.DataFrame(
                    [
                        {
                            "Rallies won": rallies_won,
                            "Actions": actions if actions > 0 else np.nan,
                        }
                    ]
                ),
                successes_col="Rallies won",
                trials_col="Actions",
                prefix="% rally won",
            ).iloc[0]
            total_row[(segment_name, "% rally won 95% CI low")] = rally_stats[
                "% rally won 95% CI low"
            ]
            total_row[(segment_name, "% rally won 95% CI high")] = rally_stats[
                "% rally won 95% CI high"
            ]

    rows.append(total_row)
    out = pd.DataFrame(rows, index=condition_order + ["Grand total"])
    out.index.name = "Condition"
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def _normalize_eval_code(value: object, known_codes: list[str]) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    code = text[0]
    if code in known_codes:
        return code
    return "OTHER"


def _normalize_player_name(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"
    return text


def _ordered_codes(values: pd.Series, preferred_order: list[str]) -> list[str]:
    observed = {str(v).strip() for v in values.dropna().tolist() if str(v).strip()}
    ordered = [code for code in preferred_order if code in observed]
    extras = sorted([code for code in observed if code not in preferred_order])
    ordered.extend(extras)
    return ordered


def _last_non_null(series: pd.Series) -> object | None:
    if series.empty:
        return None
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[-1]


def _first_int(series: pd.Series) -> int | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return int(values.iloc[0])
