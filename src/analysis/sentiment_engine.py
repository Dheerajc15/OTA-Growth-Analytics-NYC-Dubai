"""
Sentiment & Marketing Attribution Engine (Module 05)
======================================================
Data Sources: Google Places API — Dubai (#1) + YouTube Data API v3 (#5)

Business Questions:
  1. What do travelers actually SAY about Dubai hotels? (hotel review NLP)
  2. Which YouTube content themes drive the most engagement for
     NYC→Dubai travel? (marketing attribution)
  3. How does sentiment correlate with hotel performance? (rating ↔ NLP)
  4. What content gaps exist for the OTA to fill? (opportunity mapping)

Approach:
  Part A — Hotel Review Sentiment (Google Places Dubai)
    • Extract review texts from synthetic hotel data
    • VADER sentiment analysis per review
    • Aggregate sentiment by hotel, neighborhood, price tier
    • Identify sentiment drivers (topic extraction via keyword matching)

  Part B — YouTube Marketing Attribution
    • Analyze video performance by content theme
    • Engagement rate analysis (likes/views, comments/views)
    • Publish timing vs performance
    • Channel archetype benchmarking
    • Content gap identification
"""

import os
import re
import pandas as pd
import numpy as np
from collections import Counter
from typing import Optional

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    print("⚠️ vaderSentiment not installed — run: pip install vaderSentiment")

try:
    from config.settings import AB_TEST_SEED
except ImportError:
    AB_TEST_SEED = 42


# ═══════════════════════════════════════════════════════════════
# PART A: HOTEL REVIEW SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════

def extract_reviews(hotel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract individual reviews from the REVIEW_TEXTS column.

    REVIEW_TEXTS format: "review1 ||| review2 ||| review3"
    Returns one row per review with hotel metadata.
    """
    records = []
    for _, row in hotel_df.iterrows():
        review_text = row.get("REVIEW_TEXTS", "")
        if not review_text or pd.isna(review_text):
            continue

        reviews = [r.strip() for r in str(review_text).split("|||") if r.strip()]
        for j, text in enumerate(reviews):
            records.append({
                "PLACE_ID": row["PLACE_ID"],
                "HOTEL_NAME": row["NAME"],
                "MARKET": row["MARKET"],
                "NEIGHBORHOOD": row.get("NEIGHBORHOOD", ""),
                "PRICE_TIER": row.get("PRICE_TIER", "Unknown"),
                "RATING": row.get("RATING"),
                "REVIEW_INDEX": j,
                "REVIEW_TEXT": text,
            })

    df = pd.DataFrame(records)
    print(f"Extracted {len(df):,} individual reviews from "
          f"{hotel_df['PLACE_ID'].nunique()} hotels")
    return df


def analyze_review_sentiment(review_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run VADER sentiment analysis on each review.

    Adds:
      - VADER_COMPOUND: -1 (most negative) to +1 (most positive)
      - VADER_POS, VADER_NEU, VADER_NEG: component scores
      - SENTIMENT_LABEL: "Positive" / "Neutral" / "Negative"
    """
    if not HAS_VADER:
        print("⚠️ VADER not available — using fallback keyword sentiment")
        return _fallback_sentiment(review_df)

    analyzer = SentimentIntensityAnalyzer()
    df = review_df.copy()

    sentiments = df["REVIEW_TEXT"].apply(
        lambda t: analyzer.polarity_scores(str(t))
    )

    df["VADER_COMPOUND"] = sentiments.apply(lambda s: s["compound"])
    df["VADER_POS"] = sentiments.apply(lambda s: s["pos"])
    df["VADER_NEU"] = sentiments.apply(lambda s: s["neu"])
    df["VADER_NEG"] = sentiments.apply(lambda s: s["neg"])

    # Label
    df["SENTIMENT_LABEL"] = df["VADER_COMPOUND"].apply(
        lambda c: "Positive" if c >= 0.05
        else "Negative" if c <= -0.05
        else "Neutral"
    )

    print(f"\nSentiment analysis complete:")
    print(f"  Positive: {(df['SENTIMENT_LABEL']=='Positive').sum()} "
          f"({(df['SENTIMENT_LABEL']=='Positive').mean():.0%})")
    print(f"  Neutral:  {(df['SENTIMENT_LABEL']=='Neutral').sum()} "
          f"({(df['SENTIMENT_LABEL']=='Neutral').mean():.0%})")
    print(f"  Negative: {(df['SENTIMENT_LABEL']=='Negative').sum()} "
          f"({(df['SENTIMENT_LABEL']=='Negative').mean():.0%})")

    return df


def _fallback_sentiment(review_df: pd.DataFrame) -> pd.DataFrame:
    """Simple keyword-based sentiment when VADER is unavailable."""
    df = review_df.copy()

    pos_words = {
        "amazing", "stunning", "perfect", "impeccable", "best",
        "great", "love", "excellent", "beautiful", "fantastic",
        "insane", "wonderful", "good", "clean", "modern",
        "friendly", "helpful", "nice", "resort", "character",
    }
    neg_words = {
        "overpriced", "expensive", "terrible", "noise", "noisy",
        "small", "tiny", "uncomfortable", "bad", "worst",
        "dirty", "rude", "slow", "awful", "disappointing",
    }

    def _score(text):
        words = set(re.findall(r'\w+', str(text).lower()))
        pos = len(words & pos_words)
        neg = len(words & neg_words)
        total = pos + neg
        if total == 0:
            return 0.0
        return round((pos - neg) / total, 3)

    df["VADER_COMPOUND"] = df["REVIEW_TEXT"].apply(_score)
    df["VADER_POS"] = df["VADER_COMPOUND"].clip(lower=0)
    df["VADER_NEU"] = 1 - df["VADER_COMPOUND"].abs()
    df["VADER_NEG"] = (-df["VADER_COMPOUND"]).clip(lower=0)

    df["SENTIMENT_LABEL"] = df["VADER_COMPOUND"].apply(
        lambda c: "Positive" if c > 0
        else "Negative" if c < 0
        else "Neutral"
    )

    return df


# ═══════════════════════════════════════════════════════════════
# TOPIC / KEYWORD EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_sentiment_topics(review_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract key topics from reviews using keyword matching.

    Topics: location, service, price/value, room quality,
            food/dining, amenities, cleanliness, views
    """
    topic_keywords = {
        "location": [
            "location", "walkable", "close", "near", "access",
            "subway", "central", "neighborhood", "area", "distance",
        ],
        "service": [
            "service", "staff", "friendly", "helpful", "concierge",
            "reception", "attentive", "rude", "slow",
        ],
        "price_value": [
            "price", "expensive", "overpriced", "value", "worth",
            "budget", "cheap", "cost", "money", "afford",
        ],
        "room_quality": [
            "room", "bed", "bathroom", "clean", "modern",
            "small", "tiny", "spacious", "comfortable", "uncomfortable",
        ],
        "food_dining": [
            "breakfast", "restaurant", "food", "buffet", "bar",
            "dining", "eat", "rooftop", "coffee",
        ],
        "amenities": [
            "pool", "gym", "spa", "wifi", "ac", "parking",
            "lounge", "fitness", "sauna",
        ],
        "views_ambiance": [
            "view", "views", "skyline", "ocean", "stunning",
            "beautiful", "desert", "sunset", "atmosphere",
        ],
    }

    df = review_df.copy()

    for topic, keywords in topic_keywords.items():
        pattern = "|".join(keywords)
        df[f"TOPIC_{topic.upper()}"] = df["REVIEW_TEXT"].str.lower().str.contains(
            pattern, regex=True, na=False
        ).astype(int)

    # Primary topic (highest signal)
    topic_cols = [c for c in df.columns if c.startswith("TOPIC_")]
    df["PRIMARY_TOPIC"] = df[topic_cols].idxmax(axis=1).str.replace(
        "TOPIC_", "", regex=False
    ).str.lower()

    # Handle reviews with no topic match
    no_topic = df[topic_cols].sum(axis=1) == 0
    df.loc[no_topic, "PRIMARY_TOPIC"] = "general"

    print(f"\nTopic extraction:")
    print(f"  {dict(df['PRIMARY_TOPIC'].value_counts())}")

    return df


def aggregate_sentiment_by_group(
    review_df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """
    Aggregate sentiment scores by a grouping column
    (MARKET, NEIGHBORHOOD, PRICE_TIER, HOTEL_NAME, etc.)
    """
    agg = review_df.groupby(group_col).agg(
        REVIEW_COUNT=("REVIEW_TEXT", "count"),
        AVG_COMPOUND=("VADER_COMPOUND", "mean"),
        MEDIAN_COMPOUND=("VADER_COMPOUND", "median"),
        STD_COMPOUND=("VADER_COMPOUND", "std"),
        PCT_POSITIVE=("SENTIMENT_LABEL", lambda x: (x == "Positive").mean() * 100),
        PCT_NEGATIVE=("SENTIMENT_LABEL", lambda x: (x == "Negative").mean() * 100),
        PCT_NEUTRAL=("SENTIMENT_LABEL", lambda x: (x == "Neutral").mean() * 100),
    ).reset_index()

    agg = agg.round(3)
    return agg.sort_values("AVG_COMPOUND", ascending=False)


def sentiment_rating_correlation(review_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze correlation between Google rating and VADER sentiment.
    Groups by hotel and compares avg rating vs avg compound score.
    """
    hotel_agg = review_df.groupby(["PLACE_ID", "HOTEL_NAME", "MARKET"]).agg(
        GOOGLE_RATING=("RATING", "first"),
        AVG_SENTIMENT=("VADER_COMPOUND", "mean"),
        REVIEW_COUNT=("REVIEW_TEXT", "count"),
    ).reset_index()

    hotel_agg["GOOGLE_RATING"] = pd.to_numeric(
        hotel_agg["GOOGLE_RATING"], errors="coerce"
    )

    corr = hotel_agg[["GOOGLE_RATING", "AVG_SENTIMENT"]].corr().iloc[0, 1]
    print(f"\nRating ↔ Sentiment correlation: {corr:.3f}")

    # Identify mismatches (high rating but negative sentiment, or vice versa)
    hotel_agg["MISMATCH"] = "Aligned"
    hotel_agg.loc[
        (hotel_agg["GOOGLE_RATING"] >= 4.0) &
        (hotel_agg["AVG_SENTIMENT"] < 0),
        "MISMATCH",
    ] = "High Rating + Negative Sentiment"
    hotel_agg.loc[
        (hotel_agg["GOOGLE_RATING"] < 3.5) &
        (hotel_agg["AVG_SENTIMENT"] > 0.3),
        "MISMATCH",
    ] = "Low Rating + Positive Sentiment"

    mismatch_count = (hotel_agg["MISMATCH"] != "Aligned").sum()
    print(f"  Mismatched hotels: {mismatch_count}")

    return hotel_agg


# ═══════════════════════════════════════════════════════════════
# PART B: YOUTUBE MARKETING ATTRIBUTION
# ═══════════════════════════════════════════════════════════════

def prepare_youtube_data(yt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich YouTube video data with engagement metrics.
    """
    df = yt_df.copy()

    # Ensure numeric
    for col in ["VIEW_COUNT", "LIKE_COUNT", "COMMENT_COUNT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "PUBLISHED_AT" in df.columns:
        df["PUBLISHED_AT"] = pd.to_datetime(df["PUBLISHED_AT"], errors="coerce")
        df["PUBLISH_YEAR"] = df["PUBLISHED_AT"].dt.year
        df["PUBLISH_MONTH"] = df["PUBLISHED_AT"].dt.month
        df["PUBLISH_DOW"] = df["PUBLISHED_AT"].dt.day_name()

    # Engagement rates
    df["LIKE_RATE"] = (
        df["LIKE_COUNT"] / df["VIEW_COUNT"].clip(lower=1) * 100
    ).round(3)
    df["COMMENT_RATE"] = (
        df["COMMENT_COUNT"] / df["VIEW_COUNT"].clip(lower=1) * 100
    ).round(3)
    df["ENGAGEMENT_RATE"] = (
        (df["LIKE_COUNT"] + df["COMMENT_COUNT"])
        / df["VIEW_COUNT"].clip(lower=1) * 100
    ).round(3)

    # View tier
    df["VIEW_TIER"] = pd.cut(
        df["VIEW_COUNT"],
        bins=[0, 10000, 50000, 200000, 1000000, float("inf")],
        labels=["Micro (<10K)", "Small (10-50K)", "Medium (50-200K)",
                "Large (200K-1M)", "Viral (1M+)"],
    )

    # Duration bucket (if available)
    if "DURATION_MINUTES" in df.columns:
        df["DURATION_MINUTES"] = pd.to_numeric(
            df["DURATION_MINUTES"], errors="coerce"
        )
        df["DURATION_BUCKET"] = pd.cut(
            df["DURATION_MINUTES"],
            bins=[0, 5, 10, 20, 30, 120],
            labels=["Short (<5m)", "Medium (5-10m)", "Standard (10-20m)",
                    "Long (20-30m)", "Extended (30m+)"],
        )

    print(f"Prepared YouTube data: {len(df)} videos")
    print(f"  Total views: {df['VIEW_COUNT'].sum():,}")
    print(f"  Avg engagement rate: {df['ENGAGEMENT_RATE'].mean():.2f}%")
    return df


def analyze_theme_performance(yt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performance breakdown by content theme.
    """
    if "CONTENT_THEME" not in yt_df.columns:
        # Infer theme from SEARCH_QUERY if not present
        theme_map = {
            "vlog": "travel_vlog",
            "guide": "travel_guide",
            "hotel": "hotel_review",
            "things": "things_to_do",
            "budget": "budget_planning",
            "flying": "flight_review",
        }
        yt_df = yt_df.copy()
        yt_df["CONTENT_THEME"] = "other"
        for keyword, theme in theme_map.items():
            mask = yt_df["SEARCH_QUERY"].str.lower().str.contains(
                keyword, na=False
            )
            yt_df.loc[mask, "CONTENT_THEME"] = theme

    perf = yt_df.groupby("CONTENT_THEME").agg(
        VIDEO_COUNT=("VIDEO_ID", "count"),
        TOTAL_VIEWS=("VIEW_COUNT", "sum"),
        AVG_VIEWS=("VIEW_COUNT", "mean"),
        MEDIAN_VIEWS=("VIEW_COUNT", "median"),
        AVG_LIKES=("LIKE_COUNT", "mean"),
        AVG_COMMENTS=("COMMENT_COUNT", "mean"),
        AVG_LIKE_RATE=("LIKE_RATE", "mean"),
        AVG_COMMENT_RATE=("COMMENT_RATE", "mean"),
        AVG_ENGAGEMENT=("ENGAGEMENT_RATE", "mean"),
    ).reset_index()

    perf["VIEW_SHARE_PCT"] = (
        perf["TOTAL_VIEWS"] / perf["TOTAL_VIEWS"].sum() * 100
    ).round(1)

    perf = perf.sort_values("TOTAL_VIEWS", ascending=False)

    print(f"\nTheme Performance:")
    for _, row in perf.iterrows():
        print(f"  {row['CONTENT_THEME']}: {row['VIDEO_COUNT']} videos, "
              f"{row['TOTAL_VIEWS']:,.0f} views, "
              f"{row['AVG_ENGAGEMENT']:.2f}% engagement")

    return perf.round(2)


def analyze_publish_timing(yt_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze when videos are published and how timing affects performance."""
    if "PUBLISH_YEAR" not in yt_df.columns:
        return pd.DataFrame()

    timing = yt_df.groupby("PUBLISH_YEAR").agg(
        VIDEO_COUNT=("VIDEO_ID", "count"),
        TOTAL_VIEWS=("VIEW_COUNT", "sum"),
        AVG_VIEWS=("VIEW_COUNT", "mean"),
        AVG_ENGAGEMENT=("ENGAGEMENT_RATE", "mean"),
    ).reset_index()

    timing = timing.sort_values("PUBLISH_YEAR")
    return timing.round(2)


def analyze_channel_performance(yt_df: pd.DataFrame) -> pd.DataFrame:
    """Performance by channel type / archetype."""
    if "CHANNEL_TYPE" not in yt_df.columns:
        return pd.DataFrame()

    ch_perf = yt_df.groupby("CHANNEL_TYPE").agg(
        VIDEO_COUNT=("VIDEO_ID", "count"),
        TOTAL_VIEWS=("VIEW_COUNT", "sum"),
        AVG_VIEWS=("VIEW_COUNT", "mean"),
        AVG_ENGAGEMENT=("ENGAGEMENT_RATE", "mean"),
        AVG_LIKE_RATE=("LIKE_RATE", "mean"),
    ).reset_index()

    ch_perf = ch_perf.sort_values("AVG_VIEWS", ascending=False)
    return ch_perf.round(2)


def identify_content_gaps(
    theme_perf: pd.DataFrame,
    yt_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify content gaps — themes with high demand but low supply,
    or high engagement but few creators.
    """
    gaps = theme_perf.copy()

    # Supply score (normalized video count)
    max_count = gaps["VIDEO_COUNT"].max() or 1
    gaps["SUPPLY_SCORE"] = (gaps["VIDEO_COUNT"] / max_count * 100).round(1)

    # Demand score (normalized views)
    max_views = gaps["AVG_VIEWS"].max() or 1
    gaps["DEMAND_SCORE"] = (gaps["AVG_VIEWS"] / max_views * 100).round(1)

    # Engagement score
    max_eng = gaps["AVG_ENGAGEMENT"].max() or 1
    gaps["ENGAGEMENT_SCORE"] = (gaps["AVG_ENGAGEMENT"] / max_eng * 100).round(1)

    # Gap score = high demand × high engagement ÷ supply
    gaps["GAP_SCORE"] = (
        (gaps["DEMAND_SCORE"] * gaps["ENGAGEMENT_SCORE"])
        / gaps["SUPPLY_SCORE"].clip(lower=1)
    ).round(1)

    # Opportunity label
    def _label(row):
        if row["GAP_SCORE"] > 150:
            return "🔥 High Opportunity"
        elif row["GAP_SCORE"] > 80:
            return "📈 Moderate Opportunity"
        else:
            return "✅ Well-Covered"

    gaps["OPPORTUNITY"] = gaps.apply(_label, axis=1)
    gaps = gaps.sort_values("GAP_SCORE", ascending=False)

    return gaps


# ═══════════════════════════════════════════════════════════════
# OTA MARKETING RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════

def generate_marketing_recommendations(
    theme_perf: pd.DataFrame,
    content_gaps: pd.DataFrame,
    sentiment_by_market: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate actionable marketing recommendations combining
    sentiment insights + YouTube content analysis.
    """
    recs = []

    # From content gaps
    high_opp = content_gaps[
        content_gaps["OPPORTUNITY"].str.contains("High|Moderate")
    ]
    for _, row in high_opp.iterrows():
        theme = row["CONTENT_THEME"]

        strategy_map = {
            "budget_planning": {
                "action": "Create 'Dubai on $X/day' content series",
                "channel": "TikTok + YouTube Shorts",
                "rationale": "High demand, low supply — budget-conscious NYC travelers need reassurance",
            },
            "hotel_review": {
                "action": "Partner with micro-influencers for honest hotel reviews",
                "channel": "YouTube long-form + Instagram Reels",
                "rationale": "Review content builds trust for unfamiliar destination",
            },
            "things_to_do": {
                "action": "Produce 'Top 10' listicle videos with bookable links",
                "channel": "YouTube + Pinterest",
                "rationale": "Activity content has high engagement and drives tour bookings",
            },
            "flight_review": {
                "action": "Create airline comparison content (Emirates vs economy options)",
                "channel": "YouTube + blog/SEO",
                "rationale": "Flight reviews influence booking decisions and drive affiliate revenue",
            },
            "travel_vlog": {
                "action": "Sponsor authentic vlogger trips NYC→Dubai",
                "channel": "YouTube + TikTok",
                "rationale": "Vlogs build emotional connection with destination",
            },
            "travel_guide": {
                "action": "Produce comprehensive 'First Time in Dubai' guide",
                "channel": "YouTube + OTA blog",
                "rationale": "Guide content captures top-of-funnel search intent",
            },
        }

        info = strategy_map.get(theme, {
            "action": f"Explore content creation for '{theme}'",
            "channel": "YouTube",
            "rationale": "Underserved content theme with audience demand",
        })

        recs.append({
            "THEME": theme,
            "GAP_SCORE": row["GAP_SCORE"],
            "OPPORTUNITY": row["OPPORTUNITY"],
            "ACTION": info["action"],
            "CHANNEL": info["channel"],
            "RATIONALE": info["rationale"],
        })

    # From sentiment insights
    if not sentiment_by_market.empty:
        for _, row in sentiment_by_market.iterrows():
            market = row.get("MARKET", "Unknown")
            pct_neg = row.get("PCT_NEGATIVE", 0)
            if pct_neg > 20:
                recs.append({
                    "THEME": f"sentiment_{market.lower()}",
                    "GAP_SCORE": pct_neg,
                    "OPPORTUNITY": "⚠️ Sentiment Alert",
                    "ACTION": f"Address negative review themes in {market} "
                              f"hotel listings (highlight verified reviews)",
                    "CHANNEL": "OTA platform UX",
                    "RATIONALE": f"{pct_neg:.0f}% negative sentiment detected — "
                                 f"proactive response needed",
                })

    return pd.DataFrame(recs)


# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_full_sentiment_analysis(
    hotel_df: pd.DataFrame,
    youtube_df: pd.DataFrame,
) -> dict:
    """
    Run the complete M05 sentiment + marketing attribution pipeline.
    """
    print("\n" + "=" * 60)
    print("  M05: SENTIMENT & MARKETING ATTRIBUTION")
    print("=" * 60)

    # ── PART A: Hotel Review Sentiment ──
    print("\n--- Part A: Hotel Review Sentiment ---")
    reviews = extract_reviews(hotel_df)

    if reviews.empty:
        print("  No reviews found — skipping sentiment analysis")
        sentiment_reviews = pd.DataFrame()
        sentiment_by_market = pd.DataFrame()
        sentiment_by_tier = pd.DataFrame()
        sentiment_by_neighborhood = pd.DataFrame()
        rating_corr = pd.DataFrame()
    else:
        sentiment_reviews = analyze_review_sentiment(reviews)
        sentiment_reviews = extract_sentiment_topics(sentiment_reviews)

        sentiment_by_market = aggregate_sentiment_by_group(
            sentiment_reviews, "MARKET"
        )
        sentiment_by_tier = aggregate_sentiment_by_group(
            sentiment_reviews, "PRICE_TIER"
        )
        sentiment_by_neighborhood = aggregate_sentiment_by_group(
            sentiment_reviews, "NEIGHBORHOOD"
        )
        rating_corr = sentiment_rating_correlation(sentiment_reviews)

    # ── PART B: YouTube Marketing Attribution ──
    print("\n--- Part B: YouTube Marketing Attribution ---")
    yt_prepared = prepare_youtube_data(youtube_df)
    theme_perf = analyze_theme_performance(yt_prepared)
    publish_timing = analyze_publish_timing(yt_prepared)
    channel_perf = analyze_channel_performance(yt_prepared)
    content_gaps = identify_content_gaps(theme_perf, yt_prepared)

    # ── Combined Recommendations ──
    print("\n--- Marketing Recommendations ---")
    recommendations = generate_marketing_recommendations(
        theme_perf, content_gaps, sentiment_by_market
    )

    return {
        # Part A
        "reviews": sentiment_reviews,
        "sentiment_by_market": sentiment_by_market,
        "sentiment_by_tier": sentiment_by_tier,
        "sentiment_by_neighborhood": sentiment_by_neighborhood,
        "rating_correlation": rating_corr,
        # Part B
        "youtube_prepared": yt_prepared,
        "theme_performance": theme_perf,
        "publish_timing": publish_timing,
        "channel_performance": channel_perf,
        "content_gaps": content_gaps,
        # Combined
        "recommendations": recommendations,
    }


# ═══════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.data_collection.google_places import generate_synthetic_hotels
    from src.analysis.funnel_analyzer import prepare_funnel_data
    from src.data_collection.youtube_collector import generate_synthetic_youtube_data

    hotels = generate_synthetic_hotels()
    hotels = prepare_funnel_data(hotels)
    youtube = generate_synthetic_youtube_data()

    results = run_full_sentiment_analysis(hotels, youtube)
    print("\n✅ M05 Sentiment & Marketing Attribution complete")