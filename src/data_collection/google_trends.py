from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except ImportError:
    HAS_PYTRENDS = False

try:
    from config.settings import (
        TRENDS_KEYWORDS, TRENDS_TIMEFRAME, TRENDS_GEO, DATA_RAW
    )
except ImportError:
    TRENDS_KEYWORDS = ["NYC to Dubai flights", "Dubai hotels"]
    TRENDS_TIMEFRAME = "2019-01-01 2025-12-31"
    TRENDS_GEO = "US-NY"
    DATA_RAW = Path("data/raw")


def fetch_google_trends(
    keywords: Optional[list[str]] = None,
    timeframe: Optional[str] = None,
    geo: Optional[str] = None,
) -> pd.DataFrame:
    if not HAS_PYTRENDS:
        raise ImportError("pytrends not installed. pip install pytrends")

    keywords = keywords or TRENDS_KEYWORDS
    timeframe = timeframe or TRENDS_TIMEFRAME
    geo = geo or TRENDS_GEO

    pytrends = TrendReq(hl="en-US", tz=360)
    pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()

    if df.empty:
        return df

    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    df = df.reset_index().rename(columns={"date": "DATE"})
    return df

fetch_trends_data = fetch_google_trends

def save_trends_data(df: pd.DataFrame, name: str = "google_trends") -> Path:
    out_dir = DATA_RAW / "google_trends"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved → {path}")
    return path


def load_trends_data(name: str = "google_trends") -> pd.DataFrame:
    path = DATA_RAW / "google_trends" / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return pd.read_csv(path, parse_dates=["DATE"])

