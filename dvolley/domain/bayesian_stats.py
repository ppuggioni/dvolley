from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta


def beta_credible_interval(
    successes: pd.Series | np.ndarray,
    trials: pd.Series | np.ndarray,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    success_vals = np.asarray(successes, dtype=float)
    trial_vals = np.asarray(trials, dtype=float)
    failure_vals = trial_vals - success_vals

    lower = np.full(success_vals.shape, np.nan, dtype=float)
    upper = np.full(success_vals.shape, np.nan, dtype=float)

    valid = (
        np.isfinite(success_vals)
        & np.isfinite(trial_vals)
        & (trial_vals > 0)
        & (success_vals >= 0)
        & (failure_vals >= 0)
    )
    if not np.any(valid):
        return lower, upper

    alpha = (1.0 - level) / 2.0
    a = success_vals[valid] + prior_alpha
    b = failure_vals[valid] + prior_beta
    lower[valid] = beta.ppf(alpha, a, b)
    upper[valid] = beta.ppf(1.0 - alpha, a, b)
    return lower, upper


def add_beta_interval_columns(
    df: pd.DataFrame,
    *,
    successes_col: str,
    trials_col: str,
    prefix: str,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    level: float = 0.95,
) -> pd.DataFrame:
    out = df.copy()
    low, high = beta_credible_interval(
        out[successes_col],
        out[trials_col],
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        level=level,
    )
    out[f"{prefix} 95% CI low"] = low
    out[f"{prefix} 95% CI high"] = high
    return out


def format_ci_range(
    low: object,
    high: object,
    *,
    decimals: int = 2,
) -> str:
    if pd.isna(low) or pd.isna(high):
        return "-"
    return f"[{float(low):.{decimals}%}, {float(high):.{decimals}%}]"
