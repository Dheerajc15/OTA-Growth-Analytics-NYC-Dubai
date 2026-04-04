"""
Trends Preprocessing
=====================
Resample weekly Google Trends to monthly and apply cleaning.
"""

import pandas as pd


def resample_trends_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample weekly trends to monthly (mean) for merging with Aviation Edge."""
    monthly = df.resample("MS").mean().round(1)
    monthly.index.name = "DATE"
    return monthly


def clean_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure index is DatetimeIndex, drop fully-null rows,
    and forward-fill small gaps (up to 2 weeks).
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.dropna(how="all")
    out = out.ffill(limit=2)
    return out
