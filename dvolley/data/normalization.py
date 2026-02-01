from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from dvolley.config import DATE_FORMATS

def normalize_date_str(date_val) -> Optional[str]:
    if date_val is None or pd.isna(date_val):
        return None
    date_str = str(date_val).strip()
    if not date_str:
        return None
    # Strict formats only to avoid day/month swaps
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None
