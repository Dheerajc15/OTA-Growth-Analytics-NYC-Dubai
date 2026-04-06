from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DATA_SEEDS, DATA_RAW, DATA_PROCESSED
from src.preprocessing.trends import clean_trends, resample_trends_monthly
from src.preprocessing.aviation import prepare_forecast_data
from src.preprocessing.hotels import prepare_funnel_data
from src.preprocessing.youtube import prepare_youtube_data
from src.preprocessing.travelers import generate_traveler_profiles


def _load(name: str, source: str) -> pd.DataFrame:
    base = DATA_SEEDS if source == "seeds" else DATA_RAW
    p_parquet = base / f"{name}.parquet"
    p_csv = base / f"{name}.csv"

    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"Missing {name} in {base}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["seeds", "raw"], default="seeds")
    args = parser.parse_args()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    trends = clean_trends(_load("google_trends", args.source))
    trends_m = resample_trends_monthly(trends)
    trends_m.to_parquet(DATA_PROCESSED / "trends_monthly.parquet")

    capacity = _load("aviation_capacity", args.source)
    forecast = prepare_forecast_data(capacity, trends)
    forecast.to_parquet(DATA_PROCESSED / "forecast_ready.parquet", index=False)

    hotels = _load("hotels", args.source)
    hotels_p = prepare_funnel_data(hotels)
    hotels_p.to_parquet(DATA_PROCESSED / "hotels_prepared.parquet", index=False)

    youtube = _load("youtube", args.source)
    youtube_p = prepare_youtube_data(youtube)
    youtube_p.to_parquet(DATA_PROCESSED / "youtube_prepared.parquet", index=False)

    travelers = generate_traveler_profiles(hotels_p)
    travelers.to_parquet(DATA_PROCESSED / "traveler_profiles.parquet", index=False)

    print(f"✅ Processed outputs written to {DATA_PROCESSED}")


if __name__ == "__main__":
    main()