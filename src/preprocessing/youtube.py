"""
YouTube Data Preprocessing
============================
Clean and enrich YouTube video data with engagement metrics.
"""

import pandas as pd


def prepare_youtube_data(yt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich YouTube video data with engagement metrics.
    """
    df = yt_df.copy()

    for col in ["VIEW_COUNT", "LIKE_COUNT", "COMMENT_COUNT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "PUBLISHED_AT" in df.columns:
        df["PUBLISHED_AT"] = pd.to_datetime(df["PUBLISHED_AT"], errors="coerce")
        df["PUBLISH_YEAR"] = df["PUBLISHED_AT"].dt.year
        df["PUBLISH_MONTH"] = df["PUBLISHED_AT"].dt.month
        df["PUBLISH_DOW"] = df["PUBLISHED_AT"].dt.day_name()

    df["LIKE_RATE"] = (df["LIKE_COUNT"] / df["VIEW_COUNT"].clip(lower=1) * 100).round(3)
    df["COMMENT_RATE"] = (df["COMMENT_COUNT"] / df["VIEW_COUNT"].clip(lower=1) * 100).round(3)
    df["ENGAGEMENT_RATE"] = (
        (df["LIKE_COUNT"] + df["COMMENT_COUNT"]) / df["VIEW_COUNT"].clip(lower=1) * 100
    ).round(3)

    df["VIEW_TIER"] = pd.cut(
        df["VIEW_COUNT"],
        bins=[0, 10000, 50000, 200000, 1000000, float("inf")],
        labels=["Micro (<10K)", "Small (10-50K)", "Medium (50-200K)", "Large (200K-1M)", "Viral (1M+)"],
    )

    if "DURATION_MINUTES" in df.columns:
        df["DURATION_MINUTES"] = pd.to_numeric(df["DURATION_MINUTES"], errors="coerce")
        df["DURATION_BUCKET"] = pd.cut(
            df["DURATION_MINUTES"],
            bins=[0, 5, 10, 20, 30, 120],
            labels=["Short (<5m)", "Medium (5-10m)", "Standard (10-20m)", "Long (20-30m)", "Extended (30m+)"],
        )

    print(f"Prepared YouTube data: {len(df)} videos")
    print(f"  Total views: {df['VIEW_COUNT'].sum():,}")
    print(f"  Avg engagement rate: {df['ENGAGEMENT_RATE'].mean():.2f}%")
    return df
