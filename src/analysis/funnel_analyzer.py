"""
Booking Funnel Analyzer (Module 02)
=====================================
Data Sources: Google Places API — Dubai (#1) + NYC (#2)

Compares how hotel markets differ between origin (NYC) and destination (Dubai):
  - Listing density & supply-side structure
  - Price distribution & price tiers
  - Rating quality & review volume
  - Funnel stage simulation: Search → View → Compare → Book
  - Drop-off analysis: where travelers abandon the funnel

Business question: "What market-level factors cause booking funnel
leakage — and how do they differ between NYC (familiar market) vs
Dubai (unfamiliar, long-haul destination)?"
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.preprocessing.hotels import prepare_funnel_data


# ═══════════════════════════════════════════════════════════════
# MARKET COMPARISON
# ═══════════════════════════════════════════════════════════════

def compare_markets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Side-by-side comparison of Dubai vs NYC hotel markets.
    Returns DataFrame with one row per market and key metrics.
    """
    metrics = []
    for market in df["MARKET"].unique():
        mdf = df[df["MARKET"] == market]
        bookable = mdf[mdf["IS_BOOKABLE"]]

        metrics.append({
            "MARKET": market,

            # Supply
            "TOTAL_LISTINGS": len(mdf),
            "BOOKABLE_LISTINGS": len(bookable),
            "BOOKABILITY_RATE": round(len(bookable) / max(len(mdf), 1) * 100, 1),

            # Pricing
            "AVG_PRICE_LEVEL": round(mdf["PRICE_LEVEL"].mean(), 2),
            "MEDIAN_PRICE_LEVEL": mdf["PRICE_LEVEL"].median(),
            "PCT_LUXURY": round(
                (mdf["PRICE_TIER"] == "Luxury").mean() * 100, 1
            ),
            "PCT_BUDGET": round(
                (mdf["PRICE_TIER"] == "Budget").mean() * 100, 1
            ),

            # Quality
            "AVG_RATING": round(mdf["RATING"].mean(), 2),
            "MEDIAN_RATING": mdf["RATING"].median(),
            "PCT_EXCELLENT": round(
                (mdf["RATING_TIER"] == "Excellent").mean() * 100, 1
            ),

            # Engagement
            "AVG_TOTAL_RATINGS": round(mdf["TOTAL_RATINGS"].mean(), 0),
            "MEDIAN_TOTAL_RATINGS": mdf["TOTAL_RATINGS"].median(),
            "AVG_VISIBILITY": round(mdf["VISIBILITY_SCORE"].mean(), 1),

            # Photos
            "AVG_PHOTOS": round(mdf["NUM_PHOTOS"].mean(), 1),
        })

    comparison = pd.DataFrame(metrics)
    return comparison


# ═══════════════════════════════════════════════════════════════
# BOOKING FUNNEL SIMULATION
# ═══════════════════════════════════════════════════════════════

def simulate_booking_funnel(
    df: pd.DataFrame,
    market: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a 5-stage OTA booking funnel for a market.

    Stages:
      1. SEARCH   — user searches "hotels in {market}" → sees all listings
      2. VIEW     — clicks on a listing (driven by visibility score)
      3. COMPARE  — looks at details, compares 2-3 options (driven by rating)
      4. INTENT   — adds to cart / selects dates (driven by price + rating combo)
      5. BOOK     — completes booking (driven by reviews + trust signals)

    Uses MARKET-LEVEL visitor pool (10,000 total distributed by
    visibility share) and per-market median price for NaN fillna.
    """
    np.random.seed(seed)
    mdf = df[df["MARKET"] == market].copy()
    n = len(mdf)

    if n == 0:
        return pd.DataFrame()

    # Base conversion rates per stage (these differ by market)
    if market == "Dubai":
        rates = {
            "search_to_view": 0.55,       # Good photos/names attract clicks
            "view_to_compare": 0.35,       # Sticker shock → big drop
            "compare_to_intent": 0.50,     # Those who compare are serious
            "intent_to_book": 0.65,        # High intent = high conversion
        }
    else:  # NYC
        rates = {
            "search_to_view": 0.40,        # Choice paralysis — too many options
            "view_to_compare": 0.50,       # Familiar market, easier to compare
            "compare_to_intent": 0.55,     # More price-competitive
            "intent_to_book": 0.70,        # Familiar city = less booking anxiety
        }

    # Modulate rates by hotel quality
    mdf["P_VIEW"] = np.clip(
        rates["search_to_view"] * (mdf["VISIBILITY_SCORE"] / 50), 0.05, 0.95
    )
    mdf["P_COMPARE"] = np.clip(
        rates["view_to_compare"] * (mdf["RATING"].fillna(3.0) / 4.0), 0.05, 0.95
    )

    market_median_price = mdf["PRICE_LEVEL"].median()
    if pd.isna(market_median_price):
        market_median_price = 2.0
    mdf["P_INTENT"] = np.clip(
        rates["compare_to_intent"] * (1 - mdf["PRICE_LEVEL"].fillna(market_median_price) / 6),
        0.05, 0.90
    )

    mdf["P_BOOK"] = np.clip(
        rates["intent_to_book"] * (mdf["TOTAL_RATINGS"].clip(0, 5000) / 3000), 0.05, 0.90
    )

    # Market-level visitor pool
    total_visitors = 10000
    visibility_share = mdf["VISIBILITY_SCORE"] / mdf["VISIBILITY_SCORE"].sum()
    mdf["STAGE_1_SEARCH"] = (total_visitors * visibility_share).round().astype(int)

    mdf["STAGE_1_SEARCH"] = mdf["STAGE_1_SEARCH"].clip(lower=1)
    # Redistribute any rounding surplus/deficit to the top hotel
    diff = total_visitors - mdf["STAGE_1_SEARCH"].sum()
    if diff != 0:
        top_idx = mdf["VISIBILITY_SCORE"].idxmax()
        mdf.loc[top_idx, "STAGE_1_SEARCH"] += diff

    mdf["STAGE_2_VIEW"] = (mdf["STAGE_1_SEARCH"] * mdf["P_VIEW"]).astype(int)
    mdf["STAGE_3_COMPARE"] = (mdf["STAGE_2_VIEW"] * mdf["P_COMPARE"]).astype(int)
    mdf["STAGE_4_INTENT"] = (mdf["STAGE_3_COMPARE"] * mdf["P_INTENT"]).astype(int)
    mdf["STAGE_5_BOOK"] = (mdf["STAGE_4_INTENT"] * mdf["P_BOOK"]).astype(int)

    return mdf


def get_funnel_summary(funnel_df: pd.DataFrame, market: str) -> pd.DataFrame:
    """
    Aggregate funnel stages into a summary table.
    """
    stages = [
        ("1. Search", "STAGE_1_SEARCH"),
        ("2. View Listing", "STAGE_2_VIEW"),
        ("3. Compare", "STAGE_3_COMPARE"),
        ("4. Intent (Select Dates)", "STAGE_4_INTENT"),
        ("5. Book", "STAGE_5_BOOK"),
    ]

    stage_1_total = funnel_df["STAGE_1_SEARCH"].sum()
    rows = []
    prev = None
    for label, col in stages:
        total = funnel_df[col].sum()
        conversion = (total / stage_1_total * 100) if stage_1_total > 0 else 0

        if prev is not None:
            # Normal stage: compute drop-off from previous stage
            drop = prev - total
            drop_pct = round((drop / prev * 100) if prev > 0 else 0, 1)
        else:
            drop = np.nan
            drop_pct = np.nan

        rows.append({
            "STAGE": label,
            "VISITORS": total,
            "DROP_OFF": drop,
            "DROP_OFF_PCT": drop_pct,
            "OVERALL_CONVERSION_PCT": round(conversion, 2),
        })
        prev = total

    summary = pd.DataFrame(rows)
    summary["MARKET"] = market
    return summary


# ═══════════════════════════════════════════════════════════════
# DROP-OFF DIAGNOSIS
# ═══════════════════════════════════════════════════════════════

def diagnose_dropoff(
    dubai_funnel: pd.DataFrame,
    nyc_funnel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify the biggest funnel drop-off differences between markets.

    Returns actionable insights per stage.

    """
    dubai_summary = get_funnel_summary(dubai_funnel, "Dubai")
    nyc_summary = get_funnel_summary(nyc_funnel, "NYC")

    merged = dubai_summary.merge(
        nyc_summary, on="STAGE", suffixes=("_DUBAI", "_NYC")
    )

    merged["DROPOFF_GAP"] = (
        merged["DROP_OFF_PCT_DUBAI"] - merged["DROP_OFF_PCT_NYC"]
    ).round(1)

    # Positive = Dubai drops more; Negative = NYC drops more
    # Handle NaN (stage 1) gracefully
    def _classify_gap(g):
        if pd.isna(g):
            return "—"
        if g > 2:
            return "Dubai"
        if g < -2:
            return "NYC"
        return "Similar"

    merged["WORSE_MARKET"] = merged["DROPOFF_GAP"].apply(_classify_gap)

    diagnosis_map = {
        "1. Search": {
            "Dubai": "—",
            "NYC": "—",
            "Similar": "—",
            "—": "—",
        },
        "2. View Listing": {
            "Dubai": "Low click-through: unfamiliar brands, fewer recognizable chains",
            "NYC": "Choice paralysis: too many options overwhelm the traveler",
            "Similar": "Both markets face moderate click-through; personalization could help both",
            "—": "—",
        },
        "3. Compare": {
            "Dubai": "Sticker shock: Dubai prices 2-3x higher than expected",
            "NYC": "Easy to compare: traveler knows NYC neighborhoods",
            "Similar": "Comparable drop-off; consider adding side-by-side comparison tools",
            "—": "—",
        },
        "4. Intent (Select Dates)": {
            "Dubai": "Visa/logistics friction: 'Do I need a visa?', long-haul anxiety",
            "NYC": "Price sensitivity: budget travelers find NYC expensive too",
            "Similar": "Both markets lose intent-stage travelers; clearer CTAs needed",
            "—": "—",
        },
        "5. Book": {
            "Dubai": "Trust gap: unfamiliar destination, fewer reviews from US travelers",
            "NYC": "High confidence: familiar city, many friend recommendations",
            "Similar": "Similar booking friction; loyalty programs could lift both",
            "—": "—",
        },
    }

    merged["DIAGNOSIS"] = merged.apply(
        lambda r: diagnosis_map.get(r["STAGE"], {}).get(r["WORSE_MARKET"], "—"),
        axis=1,
    )

    return merged


# ═══════════════════════════════════════════════════════════════
# PRICE DISTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_price_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Price tier breakdown per market."""
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

    # Add market percentage
    market_totals = df.groupby("MARKET")["PLACE_ID"].count().to_dict()
    dist["PCT_OF_MARKET"] = dist.apply(
        lambda r: round(r["COUNT"] / market_totals.get(r["MARKET"], 1) * 100, 1),
        axis=1,
    )

    return dist.sort_values(["MARKET", "COUNT"], ascending=[True, False])


# ═══════════════════════════════════════════════════════════════
# RATING-REVIEW GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_rating_review_gap(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find hotels with high ratings but low review counts (trust gap)
    and vice versa (high reviews but mediocre ratings).

    Uses PER-MARKET medians for quadrant splits so each market's
    trust landscape is evaluated against its own baseline.
    """
    df = df.copy()
    df["TRUST_QUADRANT"] = "Unknown"

    for market in df["MARKET"].unique():
        mask = df["MARKET"] == market
        mdf = df.loc[mask]

        rating_median = mdf["RATING"].median()
        reviews_median = mdf["TOTAL_RATINGS"].median()

        conditions = [
            (mdf["RATING"] >= rating_median) & (mdf["TOTAL_RATINGS"] >= reviews_median),
            (mdf["RATING"] >= rating_median) & (mdf["TOTAL_RATINGS"] < reviews_median),
            (mdf["RATING"] < rating_median) & (mdf["TOTAL_RATINGS"] >= reviews_median),
            (mdf["RATING"] < rating_median) & (mdf["TOTAL_RATINGS"] < reviews_median),
        ]
        labels = [
            "⭐ Star Performer",      # high rating + many reviews = trust
            "🔍 Hidden Gem",           # high rating + few reviews = needs marketing
            "⚠️ Known but Risky",      # low rating + many reviews = red flag
            "❌ Low Signal",            # low rating + few reviews = avoid
        ]

        df.loc[mask, "TRUST_QUADRANT"] = np.select(
            conditions, labels, default="Unknown"
        )

    quadrant_summary = (
        df.groupby(["MARKET", "TRUST_QUADRANT"])
        .agg(COUNT=("PLACE_ID", "count"), AVG_RATING=("RATING", "mean"))
        .reset_index()
    )

    return df, quadrant_summary


# ═══════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.data_collection.google_places import generate_synthetic_hotels

    print("Funnel Analyzer — Test")
    print("=" * 40)

    hotels = generate_synthetic_hotels()
    hotels = prepare_funnel_data(hotels)

    comparison = compare_markets(hotels)
    print("\n📊 Market Comparison:")
    print(comparison.T.to_string())

    dubai_funnel = simulate_booking_funnel(hotels, "Dubai")
    nyc_funnel = simulate_booking_funnel(hotels, "NYC")

    diagnosis = diagnose_dropoff(dubai_funnel, nyc_funnel)
    print("\n📊 Drop-off Diagnosis:")
    print(diagnosis[["STAGE", "DROP_OFF_PCT_DUBAI", "DROP_OFF_PCT_NYC",
                      "WORSE_MARKET", "DIAGNOSIS"]].to_string(index=False))