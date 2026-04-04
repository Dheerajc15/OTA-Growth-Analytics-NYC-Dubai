"""
Hotel Data Preprocessing
=========================
Clean and enrich hotel data for funnel analysis.
"""

import pandas as pd
import numpy as np


def prepare_funnel_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich hotel data for funnel analysis.

    Adds:
      - PRICE_TIER: "Budget" / "Mid-Range" / "Upscale" / "Luxury"
      - RATING_TIER: "Low" / "Average" / "Good" / "Excellent"
      - VISIBILITY_SCORE: composite of ratings_count x rating x photos
        (normalized PER-MARKET to avoid cross-market distortion)
      - IS_BOOKABLE: proxy for whether a traveler would consider booking
    """
    df = df.copy()

    # ── Clean numerics ──
    df["RATING"] = pd.to_numeric(df["RATING"], errors="coerce")
    df["TOTAL_RATINGS"] = pd.to_numeric(df["TOTAL_RATINGS"], errors="coerce").fillna(0).astype(int)
    df["PRICE_LEVEL"] = pd.to_numeric(df["PRICE_LEVEL"], errors="coerce")

    if "NUM_PHOTOS" in df.columns:
        df["NUM_PHOTOS"] = pd.to_numeric(df["NUM_PHOTOS"], errors="coerce").fillna(0).astype(int)
    else:
        df["NUM_PHOTOS"] = 0

    # ── Price Tier ──
    price_map = {1: "Budget", 2: "Mid-Range", 3: "Upscale", 4: "Luxury"}
    df["PRICE_TIER"] = df["PRICE_LEVEL"].map(price_map).fillna("Unknown")

    # ── Rating Tier ──
    def _rating_tier(r):
        if pd.isna(r):
            return "Unrated"
        if r >= 4.5:
            return "Excellent"
        if r >= 4.0:
            return "Good"
        if r >= 3.5:
            return "Average"
        return "Low"

    df["RATING_TIER"] = df["RATING"].apply(_rating_tier)

    # ── Visibility Score — PER-MARKET normalization ──
    df["VISIBILITY_SCORE"] = 0.0

    for market in df["MARKET"].unique():
        mask = df["MARKET"] == market
        mdf = df.loc[mask]

        max_ratings = mdf["TOTAL_RATINGS"].quantile(0.99) or 1
        max_photos = mdf["NUM_PHOTOS"].max() or 1

        score = (
            0.50 * (mdf["TOTAL_RATINGS"] / max_ratings).clip(0, 1)
            + 0.35 * (mdf["RATING"].fillna(0) / 5.0)
            + 0.15 * (mdf["NUM_PHOTOS"] / max_photos).clip(0, 1)
        ) * 100

        df.loc[mask, "VISIBILITY_SCORE"] = score.round(1)

    # ── Bookability Proxy ──
    df["IS_BOOKABLE"] = (
        (df["RATING"].notna())
        & (df["TOTAL_RATINGS"] >= 10)
        & (df["BUSINESS_STATUS"].isin(["OPERATIONAL", ""]))
    )

    print(f"Prepared funnel data: {len(df)} hotels")
    print(f"  Bookable: {df['IS_BOOKABLE'].sum()} ({df['IS_BOOKABLE'].mean():.0%})")
    return df
