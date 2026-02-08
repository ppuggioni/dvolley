from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Hashable

import pandas as pd


def rate_cell_style(
    value: object,
    baseline: object,
    *,
    ci_low: object = None,
    ci_high: object = None,
    invert: bool = False,
    max_delta: float = 0.20,
) -> str:
    if pd.isna(value) or pd.isna(baseline):
        return ""

    delta = float(value) - float(baseline)
    if invert:
        delta = -delta
    if abs(delta) < 1e-12:
        return ""

    scale = min(abs(delta) / max_delta, 1.0)
    strength = 0.18 + (0.62 * scale)

    if pd.notna(ci_low) and pd.notna(ci_high):
        if float(ci_low) <= float(baseline) <= float(ci_high):
            strength *= 0.45

    good_rgb = (98, 176, 117)
    bad_rgb = (220, 122, 112)
    target = good_rgb if delta > 0 else bad_rgb
    r = int(255 - (255 - target[0]) * strength)
    g = int(255 - (255 - target[1]) * strength)
    b = int(255 - (255 - target[2]) * strength)
    return f"background-color: rgb({r},{g},{b});"


def build_style_matrix(
    display_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    columns: Iterable[Hashable],
    baseline_fn: Callable[[pd.DataFrame, Hashable, Hashable], object],
    ci_column_fn: Callable[[Hashable], tuple[Hashable | None, Hashable | None]] | None = None,
    invert_columns: Iterable[Hashable] | None = None,
    skip_rows: Iterable[Hashable] | None = None,
) -> pd.DataFrame:
    style = pd.DataFrame("", index=display_df.index, columns=display_df.columns)
    invert_set = set(invert_columns or [])
    skip_set = set(skip_rows or [])

    for col in columns:
        if col not in display_df.columns or col not in raw_df.columns:
            continue
        ci_low_col, ci_high_col = (None, None)
        if ci_column_fn is not None:
            ci_low_col, ci_high_col = ci_column_fn(col)
        for idx in display_df.index:
            if idx in skip_set:
                continue
            if idx not in raw_df.index:
                continue
            baseline = baseline_fn(raw_df, idx, col)
            ci_low = raw_df.at[idx, ci_low_col] if ci_low_col in raw_df.columns else None
            ci_high = raw_df.at[idx, ci_high_col] if ci_high_col in raw_df.columns else None
            style.at[idx, col] = rate_cell_style(
                raw_df.at[idx, col],
                baseline,
                ci_low=ci_low,
                ci_high=ci_high,
                invert=(col in invert_set),
            )
    return style
