from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from dvolley.domain.bayesian_stats import add_beta_interval_columns
from dvolley.domain.breakpoint_touch_analysis import build_breakpoint_touch_analysis
from dvolley.domain.conditional_breakpoint_analysis import build_conditional_breakpoint_analysis
from dvolley.domain.sideout_touch_analysis import build_sideout_touch_analysis


EVENT_ORDER = ["SERVE_ERROR", "#", "+", "!", "-", "/", "ACE", "OTHER"]
ATTACK_QUALITY_ORDER = ["#", "+", "!", "-", "/", "=", "OTHER"]
EVENT_LABELS = {
    "SERVE_ERROR": "Serve error",
    "#": "#",
    "+": "+",
    "!": "!",
    "-": "-",
    "/": "/",
    "ACE": "Ace",
    "OTHER": "Other",
}


@dataclass
class DescriptiveTouchStatsResult:
    mode: str
    rallies_df: pd.DataFrame
    summary_table: pd.DataFrame
    event_keys: list[str]
    include_by_rotation: bool
    exclude_sideout_serve_errors: bool


def build_descriptive_touch_stats(
    touches_df: pd.DataFrame,
    team_id: str,
    mode: str,
    selected_match_ids: Iterable[str] | None = None,
    *,
    include_by_rotation: bool = True,
    exclude_sideout_serve_errors: bool = False,
) -> DescriptiveTouchStatsResult:
    mode = mode.strip().lower()
    if mode not in {"sideout", "breakpoint"}:
        raise ValueError("mode must be 'sideout' or 'breakpoint'")

    selected = list(selected_match_ids) if selected_match_ids else None
    if mode == "sideout":
        rallies_df = _build_sideout_rallies_df(
            touches_df=touches_df,
            team_id=team_id,
            selected_match_ids=selected,
            exclude_sideout_serve_errors=exclude_sideout_serve_errors,
        )
    else:
        rallies_df = _build_breakpoint_rallies_df(
            touches_df=touches_df,
            team_id=team_id,
            selected_match_ids=selected,
        )

    event_keys = [e for e in EVENT_ORDER if e in set(rallies_df["event_key"].tolist())] if not rallies_df.empty else []
    summary = _build_condition_table(
        df=rallies_df,
        condition_col="event_key",
        condition_order=event_keys,
        include_by_rotation=include_by_rotation,
    )

    return DescriptiveTouchStatsResult(
        mode=mode,
        rallies_df=rallies_df,
        summary_table=summary,
        event_keys=event_keys,
        include_by_rotation=include_by_rotation,
        exclude_sideout_serve_errors=exclude_sideout_serve_errors,
    )


def build_attack_quality_drilldown_table(
    rallies_df: pd.DataFrame,
    event_key: str,
    *,
    include_by_rotation: bool = True,
) -> pd.DataFrame:
    if rallies_df.empty:
        return pd.DataFrame()

    if event_key not in set(rallies_df["event_key"].dropna().tolist()):
        return pd.DataFrame()

    subset = rallies_df[rallies_df["event_key"] == event_key].copy()
    subset = subset[subset["attack_quality"].notna()].copy()
    if subset.empty:
        return pd.DataFrame()

    observed = set(subset["attack_quality"].tolist())
    attack_order = [c for c in ATTACK_QUALITY_ORDER if c in observed]
    extra = sorted([c for c in observed if c not in ATTACK_QUALITY_ORDER])
    attack_order.extend(extra)

    return _build_condition_table(
        df=subset,
        condition_col="attack_quality",
        condition_order=attack_order,
        include_by_rotation=include_by_rotation,
    )


def get_event_display_label(event_key: str) -> str:
    return EVENT_LABELS.get(event_key, str(event_key))


def _build_sideout_rallies_df(
    touches_df: pd.DataFrame,
    team_id: str,
    selected_match_ids: list[str] | None,
    *,
    exclude_sideout_serve_errors: bool,
) -> pd.DataFrame:
    result = build_sideout_touch_analysis(
        touches_df=touches_df,
        team_id=team_id,
        selected_match_ids=selected_match_ids,
    )
    base = result.receive_rallies.copy()
    if base.empty:
        return _empty_rallies_df()

    base["event_key"] = base["class_label"].map(_map_sideout_event)
    base["rotation"] = pd.to_numeric(base["setter_position"], errors="coerce").astype("Int64")
    base["success"] = pd.to_numeric(base["sideout_point"], errors="coerce").fillna(0).astype(int)

    # By default opponent serve errors are counted as won sideouts.
    base.loc[base["event_key"] == "SERVE_ERROR", "success"] = 1

    if exclude_sideout_serve_errors:
        base = base[base["event_key"] != "SERVE_ERROR"].copy()

    base["attack_quality"] = base["first_attack_eval_code"].apply(_normalize_quality_code)
    return _finalize_rallies_df(base)


def _build_breakpoint_rallies_df(
    touches_df: pd.DataFrame,
    team_id: str,
    selected_match_ids: list[str] | None,
) -> pd.DataFrame:
    bp_result = build_breakpoint_touch_analysis(
        touches_df=touches_df,
        team_id=team_id,
        selected_match_ids=selected_match_ids,
    )
    base = bp_result.serve_rallies.copy()
    if base.empty:
        return _empty_rallies_df()

    base["event_key"] = base["class_label"].map(_map_breakpoint_event)
    base["rotation"] = pd.to_numeric(base["setter_position"], errors="coerce").astype("Int64")
    base["success"] = pd.to_numeric(base["break_point"], errors="coerce").fillna(0).astype(int)

    cond = build_conditional_breakpoint_analysis(
        touches_df=touches_df,
        team_id=team_id,
        mode="breakpoint",
        selected_match_ids=selected_match_ids,
    )
    attack_lookup = cond.rally_df[["rally_id", "first_attack_quality"]].drop_duplicates()
    base = base.merge(attack_lookup, on="rally_id", how="left")
    base["attack_quality"] = base["first_attack_quality"].apply(_normalize_quality_code)

    return _finalize_rallies_df(base)


def _finalize_rallies_df(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out = out[out["event_key"].notna()].copy()
    out["rotation"] = pd.to_numeric(out["rotation"], errors="coerce")
    out = out[out["rotation"].between(1, 6, inclusive="both")].copy()
    out["rotation"] = out["rotation"].astype(int)
    out["success"] = pd.to_numeric(out["success"], errors="coerce").fillna(0).astype(int)
    cols = ["rally_id", "event_key", "rotation", "success", "attack_quality"]
    return out[cols].drop_duplicates(subset=["rally_id"], keep="first").reset_index(drop=True)


def _empty_rallies_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["rally_id", "event_key", "rotation", "success", "attack_quality"])


def _build_condition_table(
    df: pd.DataFrame,
    condition_col: str,
    condition_order: list[str],
    *,
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
            success = int(condition_df["success"].sum()) if actions else 0
            share = actions / denominator if denominator > 0 else np.nan
            success_rate = success / actions if actions > 0 else np.nan

            row[(segment_name, "Actions")] = actions
            row[(segment_name, "% share")] = share
            row[(segment_name, "Successful")] = success
            row[(segment_name, "% successful")] = success_rate
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
            segment_stats = add_beta_interval_columns(
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
            row[(segment_name, "% successful 95% CI low")] = segment_stats[
                "% successful 95% CI low"
            ]
            row[(segment_name, "% successful 95% CI high")] = segment_stats[
                "% successful 95% CI high"
            ]
        rows.append(row)

    total_row = {}
    for segment_name, rot in segments:
        segment_df = df if rot is None else df[df["rotation"] == rot]
        denominator = len(segment_df)
        actions = int(denominator)
        success = int(segment_df["success"].sum()) if actions else 0
        share = actions / denominator if denominator > 0 else np.nan
        success_rate = success / actions if actions > 0 else np.nan

        total_row[(segment_name, "Actions")] = actions
        total_row[(segment_name, "% share")] = share
        total_row[(segment_name, "Successful")] = success
        total_row[(segment_name, "% successful")] = success_rate

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

    rows.append(total_row)
    out = pd.DataFrame(rows, index=condition_order + ["Grand total"])
    out.index.name = "Condition"
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def _map_sideout_event(class_label: object) -> str | None:
    if class_label is None:
        return None
    label = str(class_label).strip()
    if label == "OPPONENT SERVE ERROR":
        return "SERVE_ERROR"
    if label == "OPPONENT ACE":
        return "ACE"
    if label.startswith("OUR RECEPTION "):
        code = label.replace("OUR RECEPTION ", "", 1).strip()
        if code in {"#", "+", "!", "-", "/"}:
            return code
    return "OTHER"


def _map_breakpoint_event(class_label: object) -> str | None:
    if class_label is None:
        return None
    label = str(class_label).strip()
    if label == "ERRORS":
        return "SERVE_ERROR"
    if label == "ACE":
        return "ACE"
    if label.startswith("OPP RECEPTION "):
        code = label.replace("OPP RECEPTION ", "", 1).strip()
        if code in {"#", "+", "!", "-", "/"}:
            return code
    return "OTHER"


def _normalize_quality_code(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    code = text[0]
    if code in ATTACK_QUALITY_ORDER:
        return code
    return "OTHER"
