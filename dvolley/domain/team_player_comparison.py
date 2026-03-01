from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dvolley.domain.player_analysis import (
    ATTACK_QUALITY_ORDER,
    PASS_QUALITY_ORDER,
    PlayerSideoutDataset,
)


@dataclass
class TeamPlayerComparisonResult:
    total_attack_table: pd.DataFrame
    first_attack_by_pass_quality: dict[str, pd.DataFrame]
    pass_quality_table: pd.DataFrame


def build_team_player_comparison(
    dataset: PlayerSideoutDataset,
) -> TeamPlayerComparisonResult:
    total_attack_rows = dataset.team_attacks[
        dataset.team_attacks["attack_quality"].notna()
    ].copy()
    total_attack_table = _build_player_quality_table(
        total_attack_rows,
        player_col="player_name",
        quality_col="attack_quality",
        outcome_col="rally_won",
        quality_order=ATTACK_QUALITY_ORDER,
    )

    first_attack_base = dataset.sideout_rallies[
        dataset.sideout_rallies["first_attack_quality"].notna()
        & dataset.sideout_rallies["first_pass_quality"].notna()
    ].copy()
    first_attack_by_pass_quality: dict[str, pd.DataFrame] = {}
    pass_order = _ordered_codes(first_attack_base["first_pass_quality"], PASS_QUALITY_ORDER)
    for pass_quality in pass_order:
        subset = first_attack_base[first_attack_base["first_pass_quality"] == pass_quality].copy()
        table = _build_player_quality_table(
            subset,
            player_col="first_attack_player",
            quality_col="first_attack_quality",
            outcome_col="sideout_point",
            quality_order=ATTACK_QUALITY_ORDER,
        )
        if not table.empty:
            first_attack_by_pass_quality[pass_quality] = table

    pass_rows = dataset.sideout_rallies[
        dataset.sideout_rallies["first_pass_quality"].notna()
    ].copy()
    pass_quality_table = _build_player_quality_table(
        pass_rows,
        player_col="first_pass_player",
        quality_col="first_pass_quality",
        outcome_col="sideout_point",
        quality_order=PASS_QUALITY_ORDER,
    )

    return TeamPlayerComparisonResult(
        total_attack_table=total_attack_table,
        first_attack_by_pass_quality=first_attack_by_pass_quality,
        pass_quality_table=pass_quality_table,
    )


def _build_player_quality_table(
    df: pd.DataFrame,
    *,
    player_col: str,
    quality_col: str,
    outcome_col: str,
    quality_order: list[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df[[player_col, quality_col, outcome_col]].copy()
    work[player_col] = work[player_col].fillna("Unknown").astype(str)
    work[quality_col] = work[quality_col].astype(str)
    work = work[work[quality_col].str.strip() != ""].copy()
    if work.empty:
        return pd.DataFrame()

    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce").fillna(0).astype(float)
    ordered_qualities = _ordered_codes(work[quality_col], quality_order)
    if not ordered_qualities:
        return pd.DataFrame()

    player_attempts = (
        work.groupby(player_col, dropna=False)
        .size()
        .reset_index(name="attempts")
        .sort_values(["attempts", player_col], ascending=[False, True])
        .reset_index(drop=True)
    )
    player_order = player_attempts[player_col].tolist()

    rows: list[dict[tuple[str, str], float | int]] = []
    index_labels: list[str] = []

    for player_name in player_order:
        player_df = work[work[player_col] == player_name].copy()
        row = _build_row(player_df, quality_col=quality_col, outcome_col=outcome_col, ordered_qualities=ordered_qualities)
        rows.append(row)
        index_labels.append(str(player_name))

    total_row = _build_row(work, quality_col=quality_col, outcome_col=outcome_col, ordered_qualities=ordered_qualities)
    rows.append(total_row)
    index_labels.append("TOTAL")

    columns: list[tuple[str, str]] = [("Efficiency", "Score")]
    for quality in ordered_qualities:
        columns.extend(
            [
                (quality, "Count"),
                (quality, "% share"),
                (quality, "% rally won"),
            ]
        )

    out = pd.DataFrame(rows, index=index_labels)
    out = out.reindex(columns=pd.MultiIndex.from_tuples(columns))
    out.index.name = "Player"
    out.columns = pd.MultiIndex.from_tuples(columns)
    return out


def _build_row(
    df: pd.DataFrame,
    *,
    quality_col: str,
    outcome_col: str,
    ordered_qualities: list[str],
) -> dict[tuple[str, str], float | int]:
    row: dict[tuple[str, str], float | int] = {}
    total = int(len(df))
    efficiency = 0.0

    for quality in ordered_qualities:
        q_df = df[df[quality_col] == quality]
        count = int(len(q_df))
        share = (count / total) if total > 0 else np.nan
        rally_win_prob = float(q_df[outcome_col].mean()) if count > 0 else np.nan

        row[(quality, "Count")] = count
        row[(quality, "% share")] = share
        row[(quality, "% rally won")] = rally_win_prob

        if count > 0 and pd.notna(rally_win_prob):
            mapped = 2.0 * rally_win_prob - 1.0
            efficiency += share * mapped

    row[("Efficiency", "Score")] = efficiency if total > 0 else np.nan
    return row


def _ordered_codes(values: pd.Series, preferred_order: list[str]) -> list[str]:
    observed = {str(v).strip() for v in values.dropna().tolist() if str(v).strip()}
    ordered = [code for code in preferred_order if code in observed]
    extras = sorted([code for code in observed if code not in preferred_order])
    ordered.extend(extras)
    return ordered
