"""
Sentiment & Marketing Attribution 
=============================================
- review-level sentiment scoring
- topic extraction via keyword buckets
- youtube theme performance helpers
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False


def extract_reviews(hotel_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in hotel_df.iterrows():
        txt = r.get("REVIEW_TEXTS", "")
        if pd.isna(txt) or str(txt).strip() == "":
            continue
        parts = [p.strip() for p in str(txt).split("|||") if p.strip()]
        for i, p in enumerate(parts):
            rows.append({
                "PLACE_ID": r.get("PLACE_ID"),
                "HOTEL_NAME": r.get("NAME"),
                "MARKET": r.get("MARKET"),
                "NEIGHBORHOOD": r.get("NEIGHBORHOOD", ""),
                "PRICE_TIER": r.get("PRICE_TIER", "Unknown"),
                "RATING": pd.to_numeric(r.get("RATING"), errors="coerce"),
                "REVIEW_INDEX": i,
                "REVIEW_TEXT": p,
            })
    return pd.DataFrame(rows)


def analyze_review_sentiment(review_df: pd.DataFrame) -> pd.DataFrame:
    df = review_df.copy()
    if df.empty:
        return df

    if HAS_VADER:
        an = SentimentIntensityAnalyzer()
        s = df["REVIEW_TEXT"].astype(str).apply(an.polarity_scores)
        df["VADER_COMPOUND"] = s.apply(lambda x: x["compound"])
        df["VADER_POS"] = s.apply(lambda x: x["pos"])
        df["VADER_NEU"] = s.apply(lambda x: x["neu"])
        df["VADER_NEG"] = s.apply(lambda x: x["neg"])
    else:
        # fallback lexical score
        pos = {"good", "great", "amazing", "perfect", "clean", "friendly", "best", "beautiful"}
        neg = {"bad", "terrible", "overpriced", "noisy", "small", "worst", "dirty", "uncomfortable"}

        def score(t):
            w = set(re.findall(r"\w+", str(t).lower()))
            p = len(w & pos)
            n = len(w & neg)
            tot = p + n
            c = 0.0 if tot == 0 else (p - n) / tot
            return {"compound": c, "pos": max(c, 0), "neu": 1 - abs(c), "neg": max(-c, 0)}

        s = df["REVIEW_TEXT"].apply(score)
        df["VADER_COMPOUND"] = s.apply(lambda x: x["compound"])
        df["VADER_POS"] = s.apply(lambda x: x["pos"])
        df["VADER_NEU"] = s.apply(lambda x: x["neu"])
        df["VADER_NEG"] = s.apply(lambda x: x["neg"])

    df["SENTIMENT_LABEL"] = df["VADER_COMPOUND"].apply(
        lambda c: "Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral")
    )
    return df


def extract_sentiment_topics(review_df: pd.DataFrame) -> pd.DataFrame:
    df = review_df.copy()
    if df.empty:
        return df

    topic_kw = {
        "location": ["location", "near", "close", "walkable", "subway", "access"],
        "service": ["service", "staff", "friendly", "helpful", "concierge", "reception"],
        "price_value": ["price", "value", "expensive", "overpriced", "cheap", "budget", "worth"],
        "room_quality": ["room", "bed", "bathroom", "clean", "small", "spacious", "comfortable"],
        "food_dining": ["food", "breakfast", "buffet", "restaurant", "bar"],
        "amenities": ["pool", "gym", "spa", "wifi", "ac", "parking"],
        "views_ambiance": ["view", "views", "skyline", "stunning", "beautiful", "sunset"],
    }

    txt = df["REVIEW_TEXT"].astype(str).str.lower()
    topic_cols = []
    for topic, kws in topic_kw.items():
        col = f"TOPIC_{topic.upper()}"
        pattern = "|".join([re.escape(k) for k in kws])
        df[col] = txt.str.contains(pattern, regex=True, na=False).astype(int)
        topic_cols.append(col)

    df["PRIMARY_TOPIC"] = df[topic_cols].idxmax(axis=1).str.replace("TOPIC_", "", regex=False).str.lower()
    no_topic = df[topic_cols].sum(axis=1) == 0
    df.loc[no_topic, "PRIMARY_TOPIC"] = "general"
    return df


def aggregate_sentiment_by_group(review_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if review_df.empty:
        return pd.DataFrame()

    out = (
        review_df.groupby(group_col)
        .agg(
            REVIEW_COUNT=("REVIEW_TEXT", "count"),
            AVG_COMPOUND=("VADER_COMPOUND", "mean"),
            MEDIAN_COMPOUND=("VADER_COMPOUND", "median"),
            STD_COMPOUND=("VADER_COMPOUND", "std"),
            PCT_POSITIVE=("SENTIMENT_LABEL", lambda x: (x == "Positive").mean() * 100),
            PCT_NEGATIVE=("SENTIMENT_LABEL", lambda x: (x == "Negative").mean() * 100),
            PCT_NEUTRAL=("SENTIMENT_LABEL", lambda x: (x == "Neutral").mean() * 100),
        )
        .reset_index()
        .round(3)
        .sort_values("AVG_COMPOUND", ascending=False)
    )
    return out


def sentiment_rating_correlation(review_df: pd.DataFrame) -> pd.DataFrame:
    if review_df.empty:
        return pd.DataFrame()

    hotel = (
        review_df.groupby(["PLACE_ID", "HOTEL_NAME", "MARKET"])
        .agg(
            GOOGLE_RATING=("RATING", "first"),
            AVG_SENTIMENT=("VADER_COMPOUND", "mean"),
            REVIEW_COUNT=("REVIEW_TEXT", "count"),
        )
        .reset_index()
    )
    hotel["GOOGLE_RATING"] = pd.to_numeric(hotel["GOOGLE_RATING"], errors="coerce")
    hotel["MISMATCH"] = "Aligned"
    hotel.loc[(hotel["GOOGLE_RATING"] >= 4.0) & (hotel["AVG_SENTIMENT"] < 0), "MISMATCH"] = "HighRating_NegSent"
    hotel.loc[(hotel["GOOGLE_RATING"] < 3.5) & (hotel["AVG_SENTIMENT"] > 0.3), "MISMATCH"] = "LowRating_PosSent"
    return hotel