from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_funnel_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["RATING"] = pd.to_numeric(df.get("RATING"), errors="coerce")
    df["TOTAL_RATINGS"] = pd.to_numeric(df.get("TOTAL_RATINGS"), errors="coerce").fillna(0).astype(int)
    df["PRICE_LEVEL"] = pd.to_numeric(df.get("PRICE_LEVEL"), errors="coerce")

    if "NUM_PHOTOS" in df.columns:
        df["NUM_PHOTOS"] = pd.to_numeric(df["NUM_PHOTOS"], errors="coerce").fillna(0).astype(int)
    else:
        df["NUM_PHOTOS"] = 0

    price_map = {1: "Budget", 2: "Mid-Range", 3: "Upscale", 4: "Luxury"}
    df["PRICE_TIER"] = df["PRICE_LEVEL"].map(price_map).fillna("Unknown")

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

    df["VISIBILITY_SCORE"] = 0.0
    for market in df["MARKET"].dropna().unique():
        mask = df["MARKET"] == market
        mdf = df.loc[mask]

        max_ratings = float(mdf["TOTAL_RATINGS"].quantile(0.99) or 1.0)
        max_photos = float(mdf["NUM_PHOTOS"].max() or 1.0)

        score = (
            0.50 * (mdf["TOTAL_RATINGS"] / max_ratings).clip(0, 1)
            + 0.35 * (mdf["RATING"].fillna(0) / 5.0)
            + 0.15 * (mdf["NUM_PHOTOS"] / max_photos).clip(0, 1)
        ) * 100.0

        df.loc[mask, "VISIBILITY_SCORE"] = score.round(1)

    df["IS_BOOKABLE"] = (
        df["RATING"].notna()
        & (df["TOTAL_RATINGS"] >= 10)
        & (df.get("BUSINESS_STATUS", "").fillna("").isin(["OPERATIONAL", ""]))
    )

    return df


def compare_markets(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market in df["MARKET"].dropna().unique():
        mdf = df[df["MARKET"] == market]
        bookable = mdf[mdf["IS_BOOKABLE"]]

        rows.append({
            "MARKET": market,
            "TOTAL_LISTINGS": len(mdf),
            "BOOKABLE_LISTINGS": len(bookable),
            "BOOKABILITY_RATE": round(len(bookable) / max(len(mdf), 1) * 100, 1),
            "AVG_PRICE_LEVEL": round(float(mdf["PRICE_LEVEL"].mean()), 2),
            "MEDIAN_PRICE_LEVEL": float(mdf["PRICE_LEVEL"].median()),
            "PCT_LUXURY": round((mdf["PRICE_TIER"] == "Luxury").mean() * 100, 1),
            "PCT_BUDGET": round((mdf["PRICE_TIER"] == "Budget").mean() * 100, 1),
            "AVG_RATING": round(float(mdf["RATING"].mean()), 2),
            "MEDIAN_RATING": float(mdf["RATING"].median()),
            "PCT_EXCELLENT": round((mdf["RATING_TIER"] == "Excellent").mean() * 100, 1),
            "AVG_TOTAL_RATINGS": round(float(mdf["TOTAL_RATINGS"].mean()), 0),
            "MEDIAN_TOTAL_RATINGS": float(mdf["TOTAL_RATINGS"].median()),
            "AVG_VISIBILITY": round(float(mdf["VISIBILITY_SCORE"].mean()), 1),
            "AVG_PHOTOS": round(float(mdf["NUM_PHOTOS"].mean()), 1),
        })
    return pd.DataFrame(rows)


def simulate_booking_funnel(df: pd.DataFrame, market: str, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    mdf = df[df["MARKET"] == market].copy()
    if mdf.empty:
        return pd.DataFrame()

    rates = (
        {"search_to_view": 0.55, "view_to_compare": 0.35, "compare_to_intent": 0.50, "intent_to_book": 0.65}
        if market == "Dubai"
        else {"search_to_view": 0.40, "view_to_compare": 0.50, "compare_to_intent": 0.55, "intent_to_book": 0.70}
    )

    mdf["P_VIEW"] = np.clip(rates["search_to_view"] * (mdf["VISIBILITY_SCORE"] / 50), 0.05, 0.95)
    mdf["P_COMPARE"] = np.clip(rates["view_to_compare"] * (mdf["RATING"].fillna(3.0) / 4.0), 0.05, 0.95)

    market_median_price = mdf["PRICE_LEVEL"].median()
    if pd.isna(market_median_price):
        market_median_price = 2.0

    mdf["P_INTENT"] = np.clip(
        rates["compare_to_intent"] * (1 - mdf["PRICE_LEVEL"].fillna(market_median_price) / 6),
        0.05, 0.90
    )
    mdf["P_BOOK"] = np.clip(
        rates["intent_to_book"] * (mdf["TOTAL_RATINGS"].clip(0, 5000) / 3000),
        0.05, 0.90
    )

    total_visitors = 10000
    denom = float(mdf["VISIBILITY_SCORE"].sum())
    if denom <= 0:
        visibility_share = np.full(len(mdf), 1 / len(mdf))
    else:
        visibility_share = (mdf["VISIBILITY_SCORE"] / denom).values

    mdf["STAGE_1_SEARCH"] = (total_visitors * visibility_share).round().astype(int).clip(min=1)
    diff = total_visitors - int(mdf["STAGE_1_SEARCH"].sum())
    if diff != 0:
        mdf.loc[mdf["VISIBILITY_SCORE"].idxmax(), "STAGE_1_SEARCH"] += diff

    mdf["STAGE_2_VIEW"] = (mdf["STAGE_1_SEARCH"] * mdf["P_VIEW"]).astype(int)
    mdf["STAGE_3_COMPARE"] = (mdf["STAGE_2_VIEW"] * mdf["P_COMPARE"]).astype(int)
    mdf["STAGE_4_INTENT"] = (mdf["STAGE_3_COMPARE"] * mdf["P_INTENT"]).astype(int)
    mdf["STAGE_5_BOOK"] = (mdf["STAGE_4_INTENT"] * mdf["P_BOOK"]).astype(int)

    return mdf


def get_funnel_summary(funnel_df: pd.DataFrame, market: str) -> pd.DataFrame:
    stages = [
        ("1. Search", "STAGE_1_SEARCH"),
        ("2. View Listing", "STAGE_2_VIEW"),
        ("3. Compare", "STAGE_3_COMPARE"),
        ("4. Intent (Select Dates)", "STAGE_4_INTENT"),
        ("5. Book", "STAGE_5_BOOK"),
    ]

    s1 = funnel_df["STAGE_1_SEARCH"].sum()
    rows = []
    prev = None
    for label, col in stages:
        total = funnel_df[col].sum()
        conversion = (total / s1 * 100) if s1 > 0 else 0.0
        if prev is None:
            drop, drop_pct = np.nan, np.nan
        else:
            drop = prev - total
            drop_pct = round((drop / prev * 100) if prev > 0 else 0, 1)

        rows.append({
            "STAGE": label,
            "VISITORS": int(total),
            "DROP_OFF": drop,
            "DROP_OFF_PCT": drop_pct,
            "OVERALL_CONVERSION_PCT": round(conversion, 2),
            "MARKET": market,
        })
        prev = total

    return pd.DataFrame(rows)


def diagnose_dropoff(dubai_funnel: pd.DataFrame, nyc_funnel: pd.DataFrame) -> pd.DataFrame:
    dubai_summary = get_funnel_summary(dubai_funnel, "Dubai")
    nyc_summary = get_funnel_summary(nyc_funnel, "NYC")

    merged = dubai_summary.merge(nyc_summary, on="STAGE", suffixes=("_DUBAI", "_NYC"))
    merged["DROPOFF_GAP"] = (merged["DROP_OFF_PCT_DUBAI"] - merged["DROP_OFF_PCT_NYC"]).round(1)

    def classify(g):
        if pd.isna(g):
            return "—"
        if g > 2:
            return "Dubai"
        if g < -2:
            return "NYC"
        return "Similar"

    merged["WORSE_MARKET"] = merged["DROPOFF_GAP"].apply(classify)
    return merged


def analyze_price_distribution(df: pd.DataFrame) -> pd.DataFrame:
    dist = (
        df.groupby(["MARKET", "PRICE_TIER"])
        .agg(
            COUNT=("PLACE_ID", "count"),
            AVG_RATING=("RATING", "mean"),
            AVG_REVIEWS=("TOTAL_RATINGS", "mean"),
            AVG_VISIBILITY=("VISIBILITY_SCORE", "mean"),
        )
        .reset_index()
    )
    totals = df.groupby("MARKET")["PLACE_ID"].count().to_dict()
    dist["PCT_OF_MARKET"] = dist.apply(
        lambda r: round(r["COUNT"] / totals.get(r["MARKET"], 1) * 100, 1), axis=1
    )
    return dist.sort_values(["MARKET", "COUNT"], ascending=[True, False])


def analyze_rating_review_gap(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["TRUST_QUADRANT"] = "Unknown"

    for market in out["MARKET"].dropna().unique():
        mdf = out[out["MARKET"] == market]
        rm = mdf["RATING"].median()
        vm = mdf["TOTAL_RATINGS"].median()

        conds = [
            (mdf["RATING"] >= rm) & (mdf["TOTAL_RATINGS"] >= vm),
            (mdf["RATING"] >= rm) & (mdf["TOTAL_RATINGS"] < vm),
            (mdf["RATING"] < rm) & (mdf["TOTAL_RATINGS"] >= vm),
            (mdf["RATING"] < rm) & (mdf["TOTAL_RATINGS"] < vm),
        ]
        labels = ["⭐ Star Performer", "🔍 Hidden Gem", "⚠️ Known but Risky", "❌ Low Signal"]

        out.loc[out["MARKET"] == market, "TRUST_QUADRANT"] = np.select(conds, labels, default="Unknown")

    summary = (
        out.groupby(["MARKET", "TRUST_QUADRANT"])
        .agg(COUNT=("PLACE_ID", "count"), AVG_RATING=("RATING", "mean"))
        .reset_index()
    )
    return out, summary