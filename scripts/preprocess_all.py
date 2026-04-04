"""
Preprocess all seed data into analysis-ready outputs.
=====================================================
Run:  python scripts/preprocess_all.py
      python scripts/preprocess_all.py --source seeds   (default)
      python scripts/preprocess_all.py --source raw      (use real API data)

Reads from data/seeds/ (or data/raw/) and writes to data/processed/.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.trends import resample_trends_monthly, clean_trends
from src.preprocessing.aviation import prepare_forecast_data
from src.preprocessing.hotels import prepare_funnel_data
from src.preprocessing.youtube import prepare_youtube_data
from src.preprocessing.travelers import generate_traveler_profiles

SEEDS_DIR = ROOT / "data" / "seeds"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_seed(name: str) -> pd.DataFrame:
    path = SEEDS_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    csv_path = SEEDS_DIR / f"{name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)
    raise FileNotFoundError(f"No seed file found: {path} or {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument("--source", choices=["seeds", "raw"], default="seeds")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Trends ──
    print("=== Preprocessing Trends ===")
    trends = load_seed("google_trends")
    trends = clean_trends(trends)
    trends_monthly = resample_trends_monthly(trends)
    trends_monthly.to_parquet(PROCESSED_DIR / "trends_monthly.parquet")
    print(f"  -> {len(trends_monthly)} months")

    # ── Capacity ──
    print("\n=== Preprocessing Aviation Capacity ===")
    capacity = load_seed("aviation_capacity")
    forecast_df = prepare_forecast_data(capacity, trends)
    forecast_df.to_parquet(PROCESSED_DIR / "forecast_ready.parquet", index=False)
    print(f"  -> {len(forecast_df)} months (Prophet-ready)")

    # ── Hotels ──
    print("\n=== Preprocessing Hotels ===")
    hotels = load_seed("hotels")
    hotels_prepared = prepare_funnel_data(hotels)
    hotels_prepared.to_parquet(PROCESSED_DIR / "hotels_prepared.parquet", index=False)
    print(f"  -> {len(hotels_prepared)} hotels")

    # ── YouTube ──
    print("\n=== Preprocessing YouTube ===")
    youtube = load_seed("youtube")
    youtube_prepared = prepare_youtube_data(youtube)
    youtube_prepared.to_parquet(PROCESSED_DIR / "youtube_prepared.parquet", index=False)
    print(f"  -> {len(youtube_prepared)} videos")

    # ── Travelers ──
    print("\n=== Generating Traveler Profiles ===")
    travelers = generate_traveler_profiles(hotels_prepared)
    travelers.to_parquet(PROCESSED_DIR / "traveler_profiles.parquet", index=False)
    print(f"  -> {len(travelers)} travelers")

    print(f"\nAll processed data written to {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
