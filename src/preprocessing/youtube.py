from __future__ import annotations

import pandas as pd


def prepare_youtube_data(yt_df: pd.DataFrame) -> pd.DataFrame:
    df = yt_df.copy()

    for col in ["VIEW_COUNT", "LIKE_COUNT", "COMMENT_COUNT"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "PUBLISHED_AT" in df.columns:
        df["PUBLISHED_AT"] = pd.to_datetime(df["PUBLISHED_AT"], errors="coerce")
        df["PUBLISH_YEAR"] = df["PUBLISHED_AT"].dt.year
        df["PUBLISH_MONTH"] = df["PUBLISHED_AT"].dt.month
        df["PUBLISH_DOW"] = df["PUBLISHED_AT"].dt.day_name()

    denom = df["VIEW_COUNT"].clip(lower=1)
    df["LIKE_RATE"] = (df["LIKE_COUNT"] / denom * 100).round(3)
    df["COMMENT_RATE"] = (df["COMMENT_COUNT"] / denom * 100).round(3)
    df["ENGAGEMENT_RATE"] = ((df["LIKE_COUNT"] + df["COMMENT_COUNT"]) / denom * 100).round(3)

    if "DURATION_MINUTES" in df.columns:
        df["DURATION_MINUTES"] = pd.to_numeric(df["DURATION_MINUTES"], errors="coerce")

    return df