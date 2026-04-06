from __future__ import annotations

import pandas as pd


def clean_trends(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "DATE" in out.columns:
        out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")
        out = out.dropna(subset=["DATE"]).set_index("DATE")
    else:
        out.index = pd.to_datetime(out.index, errors="coerce")

    out = out.dropna(how="all")
    out = out.ffill(limit=2)
    return out


def resample_trends_monthly(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_trends(df)
    monthly = out.resample("MS").mean().round(1)
    monthly.index.name = "DATE"
    return monthly