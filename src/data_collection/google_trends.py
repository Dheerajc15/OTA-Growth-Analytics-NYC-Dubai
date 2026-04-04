"""
Google Trends Data Collector (Data Source #3)
==============================================
Used in: M01 (Demand Forecasting), M06 (Visa & Friction Analysis)

Tracks search interest for NYC -> Dubai travel keywords from New York State.
No API key required — pytrends scrapes Google Trends directly.

Keywords tracked:
  - "NYC to Dubai flights"       -> direct booking intent
  - "Dubai hotels"               -> accommodation research
  - "Dubai visa"                 -> friction/planning signal
  - "Dubai tourism"              -> general awareness
  - "cheap flights to Dubai"     -> price-sensitive segment
"""

import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import time
from pathlib import Path
from typing import Optional

try:
    from config.settings import TRENDS_KEYWORDS, TRENDS_TIMEFRAME, TRENDS_GEO, DATA_RAW
except ImportError:
    TRENDS_KEYWORDS = [
        "NYC to Dubai flights", "Dubai hotels", "Dubai visa",
        "Dubai tourism", "cheap flights to Dubai",
    ]
    TRENDS_TIMEFRAME = "2019-01-01 2025-12-31"
    TRENDS_GEO = "US-NY"
    DATA_RAW = Path("data/raw")


# ═══════════════════════════════════════════════════════════════
# FETCH LIVE DATA
# ═══════════════════════════════════════════════════════════════

def fetch_google_trends(
    keywords: Optional[list[str]] = None,
    timeframe: Optional[str] = None,
    geo: Optional[str] = None,
    wait_seconds: float = 3.0,
) -> pd.DataFrame:
    """
    Fetch Google Trends interest-over-time data.

    Returns
    -------
    pd.DataFrame — index=weekly dates, columns=keywords, values=0-100
    """
    keywords = keywords or TRENDS_KEYWORDS
    timeframe = timeframe or TRENDS_TIMEFRAME
    geo = geo or TRENDS_GEO

    print(f"Fetching Google Trends for {len(keywords)} keywords...")
    print(f"  Geo: {geo}  |  Timeframe: {timeframe}")

    pytrends = TrendReq(hl="en-US", tz=300, retries=3, backoff_factor=1.0)

    all_data = []
    for i in range(0, len(keywords), 5):
        batch = keywords[i:i + 5]
        print(f"  Batch {i // 5 + 1}: {batch}")

        try:
            pytrends.build_payload(kw_list=batch, timeframe=timeframe, geo=geo)
            data = pytrends.interest_over_time()

            if not data.empty:
                data = data.drop(columns=["isPartial"], errors="ignore")
                all_data.append(data)
                print(f"    Got {len(data)} weekly data points")
            else:
                print(f"    Empty result for batch")
        except Exception as e:
            print(f"    Error: {e}")

        if i + 5 < len(keywords):
            time.sleep(wait_seconds)

    if not all_data:
        print("No data retrieved.")
        return pd.DataFrame()

    result = pd.concat(all_data, axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    print(f"Trends: {result.shape[0]} weeks x {result.shape[1]} keywords")
    return result


# ═══════════════════════════════════════════════════════════════
# SPIKE DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_search_spikes(
    df: pd.DataFrame,
    keyword: str,
    threshold_std: float = 1.5,
    rolling_window: int = 12,
) -> pd.DataFrame:
    """
    Find weeks where search interest spikes above rolling_mean + threshold x rolling_std.
    """
    if keyword not in df.columns:
        raise ValueError(f"'{keyword}' not in columns: {list(df.columns)}")

    series = df[keyword]
    rolling_mean = series.rolling(window=rolling_window, min_periods=4).mean()
    rolling_std = series.rolling(window=rolling_window, min_periods=4).std()

    spike_mask = series > (rolling_mean + threshold_std * rolling_std)
    spikes = df[spike_mask].copy()

    if spikes.empty:
        print(f"No spikes for '{keyword}' at {threshold_std} sigma")
        return pd.DataFrame()

    spikes["ROLLING_MEAN"] = rolling_mean[spike_mask]
    spikes["SPIKE_MAGNITUDE"] = series[spike_mask] / rolling_mean[spike_mask]
    spikes["KEYWORD"] = keyword

    print(f"Found {len(spikes)} spikes for '{keyword}'")
    return spikes


# ═══════════════════════════════════════════════════════════════
# SEASONAL PATTERN EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_seasonal_patterns(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """Monthly aggregation to reveal seasonal peaks/dips."""
    if keyword not in df.columns:
        raise ValueError(f"'{keyword}' not in columns")

    monthly = df[[keyword]].copy()
    monthly["month"] = monthly.index.month
    monthly["month_name"] = monthly.index.strftime("%B")

    seasonal = (
        monthly.groupby(["month", "month_name"])[keyword]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .sort_values("month")
    )
    seasonal.columns = [
        "MONTH_NUM", "MONTH_NAME", "AVG_INTEREST",
        "STD_INTEREST", "MIN_INTEREST", "MAX_INTEREST", "NUM_WEEKS",
    ]
    overall_mean = seasonal["AVG_INTEREST"].mean()
    seasonal["IS_PEAK"] = seasonal["AVG_INTEREST"] > overall_mean
    seasonal["PCT_ABOVE_MEAN"] = (
        (seasonal["AVG_INTEREST"] - overall_mean) / overall_mean * 100
    ).round(1)

    return seasonal


# ═══════════════════════════════════════════════════════════════
# SAVE / LOAD
# ═══════════════════════════════════════════════════════════════

def save_trends_data(df: pd.DataFrame, filename: str = "google_trends_nyc_dubai.csv") -> Path:
    output_dir = DATA_RAW / "google_trends"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.to_csv(path)
    print(f"Saved -> {path}")
    return path


def load_saved_trends(filename: str = "google_trends_nyc_dubai.csv") -> pd.DataFrame:
    path = DATA_RAW / "google_trends" / filename
    if not path.exists():
        raise FileNotFoundError(f"No saved trends at: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ═══════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Google Trends Collector — Test")
    print("-" * 40)
    trends = fetch_google_trends()
    if not trends.empty:
        save_trends_data(trends)
        for kw in trends.columns[:2]:
            s = extract_seasonal_patterns(trends, kw)
            peaks = s[s["IS_PEAK"]]["MONTH_NAME"].tolist()
            print(f"  '{kw}' peaks: {', '.join(peaks)}")
    else:
        print("No data retrieved. Set up pytrends or use: python scripts/generate_seeds.py")
