"""
YouTube Data API v3 Collector (Data Source #5)
================================================
Used in: M05 (Sentiment & Marketing Attribution)

Fetches YouTube video metadata for NYC→Dubai travel content:
  - Search by queries ("NYC to Dubai travel vlog", etc.)
  - Video statistics: views, likes, comments
  - Channel metadata
  - Synthetic fallback when API key unavailable
"""

import os
import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

try:
    from config.settings import (
        GOOGLE_CLOUD_API_KEY, YOUTUBE_SEARCH_QUERIES,
        YOUTUBE_MAX_RESULTS_PER_QUERY, DATA_RAW, AB_TEST_SEED,
    )
except ImportError:
    GOOGLE_CLOUD_API_KEY = None
    YOUTUBE_SEARCH_QUERIES = [
        "NYC to Dubai travel vlog",
        "Dubai travel guide from New York",
        "flying to Dubai from JFK",
        "Dubai hotel review tourist",
        "Dubai things to do American tourist",
        "Dubai trip budget from USA",
    ]
    YOUTUBE_MAX_RESULTS_PER_QUERY = 50
    DATA_RAW = Path("data/raw")
    AB_TEST_SEED = 42

# Try to import google-api-python-client
try:
    from googleapiclient.discovery import build as yt_build
    HAS_YT_CLIENT = True
except ImportError:
    HAS_YT_CLIENT = False


# ═══════════════════════════════════════════════════════════════
# LIVE API — YOUTUBE SEARCH + STATISTICS
# ═══════════════════════════════════════════════════════════════

def _get_youtube_service():
    """Build authenticated YouTube API service."""
    if not GOOGLE_CLOUD_API_KEY:
        print("⚠️ GOOGLE_CLOUD_API_KEY not set — cannot use YouTube API")
        return None
    if not HAS_YT_CLIENT:
        print("⚠️ google-api-python-client not installed")
        print("   pip install google-api-python-client")
        return None
    return yt_build("youtube", "v3", developerKey=GOOGLE_CLOUD_API_KEY)


def search_youtube_videos(
    query: str,
    max_results: int = 50,
    order: str = "relevance",
    published_after: Optional[str] = None,
) -> list[dict]:
    """
    Search YouTube for videos matching a query.

    Parameters
    ----------
    query : search terms
    max_results : up to 50 per API call
    order : "relevance", "date", "viewCount", "rating"
    published_after : ISO 8601 date (e.g. "2020-01-01T00:00:00Z")

    Returns list of video metadata dicts.
    """
    youtube = _get_youtube_service()
    if not youtube:
        return []

    params = {
        "q": query,
        "part": "snippet",
        "type": "video",
        "maxResults": min(max_results, 50),
        "order": order,
    }
    if published_after:
        params["publishedAfter"] = published_after

    try:
        response = youtube.search().list(**params).execute()
    except Exception as e:
        print(f"    YouTube API error: {e}")
        return []

    videos = []
    for item in response.get("items", []):
        snippet = item.get("snippet", {})
        videos.append({
            "VIDEO_ID": item["id"]["videoId"],
            "TITLE": snippet.get("title", ""),
            "DESCRIPTION": snippet.get("description", ""),
            "CHANNEL_TITLE": snippet.get("channelTitle", ""),
            "CHANNEL_ID": snippet.get("channelId", ""),
            "PUBLISHED_AT": snippet.get("publishedAt", ""),
            "SEARCH_QUERY": query,
        })

    print(f"    '{query}': {len(videos)} videos found")
    return videos


def get_video_statistics(video_ids: list[str]) -> dict:
    """
    Fetch view/like/comment counts for a batch of video IDs.
    API allows up to 50 IDs per request.
    """
    youtube = _get_youtube_service()
    if not youtube:
        return {}

    stats = {}
    # Process in batches of 50
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        try:
            response = youtube.videos().list(
                part="statistics,contentDetails",
                id=",".join(batch),
            ).execute()
        except Exception as e:
            print(f"    Stats API error: {e}")
            continue

        for item in response.get("items", []):
            s = item.get("statistics", {})
            cd = item.get("contentDetails", {})
            stats[item["id"]] = {
                "VIEW_COUNT": int(s.get("viewCount", 0)),
                "LIKE_COUNT": int(s.get("likeCount", 0)),
                "COMMENT_COUNT": int(s.get("commentCount", 0)),
                "DURATION": cd.get("duration", ""),
            }
        time.sleep(0.2)

    return stats


def fetch_youtube_data(
    queries: list[str] = None,
    max_per_query: int = None,
) -> pd.DataFrame:
    """
    Full pipeline: search all queries → fetch stats → return DataFrame.
    """
    queries = queries or YOUTUBE_SEARCH_QUERIES
    max_per_query = max_per_query or YOUTUBE_MAX_RESULTS_PER_QUERY

    print(f"\n{'='*60}")
    print(f"  YouTube Data Collection")
    print(f"  Queries: {len(queries)} | Max per query: {max_per_query}")
    print(f"{'='*60}")

    all_videos = {}
    for query in queries:
        results = search_youtube_videos(query, max_results=max_per_query)
        for v in results:
            vid = v["VIDEO_ID"]
            if vid not in all_videos:
                all_videos[vid] = v
        time.sleep(0.5)

    if not all_videos:
        print("  No videos found — returning empty DataFrame")
        return pd.DataFrame()

    print(f"\n  Unique videos: {len(all_videos)}")
    print(f"  Fetching statistics...")

    stats = get_video_statistics(list(all_videos.keys()))
    for vid, s in stats.items():
        if vid in all_videos:
            all_videos[vid].update(s)

    df = pd.DataFrame(all_videos.values())
    print(f"  ✅ {len(df)} videos with statistics")
    return df


# ═══════════════════════════════════════════════════════════════
# SYNTHETIC YOUTUBE DATA (offline development)
# ═══════════════════════════════════════════════════════════════

def generate_synthetic_youtube_data(
    n_videos: int = 200,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate realistic synthetic YouTube video data for
    NYC→Dubai travel content analysis.

    Simulates real YouTube patterns:
      - View counts follow power-law (few viral, many niche)
      - Like/view ratio ~3-5% for travel content
      - Comment/view ratio ~0.3-1%
      - Title patterns match real travel vlog conventions
      - Published dates span 2019-2025
    """
    seed = seed or AB_TEST_SEED
    rng = np.random.RandomState(seed)

    # ── Content themes (maps to search queries) ──
    themes = {
        "travel_vlog": {
            "weight": 0.25,
            "titles": [
                "NYC to Dubai FIRST CLASS on Emirates A380!",
                "I flew from JFK to Dubai for $400 (here's how)",
                "48 Hours in Dubai - NYC Girl's First Time!",
                "Dubai Travel Vlog 2024 | From New York to the Desert",
                "Flying Business Class NYC→Dubai | Is it worth $3,800?",
                "My INSANE Dubai Trip from New York City",
                "JFK to DXB: 14 Hours on Emirates (Full Review)",
                "NYC���Dubai: The Ultimate Travel Day Vlog",
            ],
            "avg_views": 150000, "view_std": 200000,
            "engagement_mult": 1.2,
        },
        "travel_guide": {
            "weight": 0.20,
            "titles": [
                "Dubai Travel Guide 2024: Everything You Need to Know",
                "First Time in Dubai? Watch This BEFORE You Go",
                "Dubai on a Budget: Tips from a New Yorker",
                "Top 20 Things to Do in Dubai (from an American)",
                "Dubai Travel Guide for US Citizens | Visa, Money, Safety",
                "What I Wish I Knew Before Going to Dubai from NYC",
            ],
            "avg_views": 250000, "view_std": 300000,
            "engagement_mult": 1.0,
        },
        "hotel_review": {
            "weight": 0.20,
            "titles": [
                "Atlantis the Palm Dubai - Honest Review (Worth $500/night?)",
                "I Stayed at the CHEAPEST Hotel in Dubai Marina",
                "Burj Al Arab vs Atlantis: Which Dubai Hotel is Better?",
                "Dubai Hotel Room Tour: Budget vs Luxury Comparison",
                "$50 vs $5,000 Hotel in Dubai - Is Luxury Worth It?",
                "Best Hotels in Dubai for American Tourists (2024)",
            ],
            "avg_views": 100000, "view_std": 150000,
            "engagement_mult": 1.1,
        },
        "things_to_do": {
            "weight": 0.15,
            "titles": [
                "Dubai Desert Safari - Is It a Tourist Trap?",
                "Top 10 FREE Things to Do in Dubai",
                "Dubai Mall Tour: World's Largest Shopping Mall!",
                "Trying Street Food in Old Dubai (Deira Market)",
                "Dubai Nightlife Guide for American Tourists",
                "Most UNDERRATED Things to Do in Dubai",
            ],
            "avg_views": 180000, "view_std": 250000,
            "engagement_mult": 1.3,
        },
        "budget_planning": {
            "weight": 0.10,
            "titles": [
                "How Much Does a Dubai Trip ACTUALLY Cost from NYC?",
                "Dubai Trip Budget Breakdown: 7 Days, $2,500 Total",
                "Cheap Flights from NYC to Dubai: My Strategy",
                "Is Dubai EXPENSIVE? Real Costs for Americans",
                "How to Visit Dubai on a NYC Budget",
            ],
            "avg_views": 80000, "view_std": 120000,
            "engagement_mult": 0.9,
        },
        "flight_review": {
            "weight": 0.10,
            "titles": [
                "Emirates vs Etihad: Which is Better from JFK?",
                "JFK to Dubai: Economy Class Honest Review",
                "Flying to Dubai from New York: What to Expect",
                "Emirates A380 Business Class Review (JFK→DXB)",
                "Best Airlines NYC to Dubai Ranked",
            ],
            "avg_views": 200000, "view_std": 280000,
            "engagement_mult": 1.15,
        },
    }

    # ── Channel archetypes ──
    channels = [
        {"name": "WanderLuxe NYC", "subs": 850000, "type": "luxury"},
        {"name": "Budget Backpacker", "subs": 420000, "type": "budget"},
        {"name": "The Points Guy Travel", "subs": 1200000, "type": "points"},
        {"name": "NYC Travel Diaries", "subs": 350000, "type": "local"},
        {"name": "Kara and Nate", "subs": 3500000, "type": "couple"},
        {"name": "Mark Wiens (Food)", "subs": 9500000, "type": "food"},
        {"name": "Lost LeBlanc", "subs": 2800000, "type": "cinematic"},
        {"name": "Hey Nadine", "subs": 750000, "type": "solo_female"},
        {"name": "Wolters World", "subs": 1600000, "type": "tips"},
        {"name": "Flying The Nest", "subs": 900000, "type": "family"},
        {"name": "Nomadic Matt", "subs": 500000, "type": "budget"},
        {"name": "Sorelle Amore", "subs": 1100000, "type": "lifestyle"},
        {"name": "Travel Tips by Dheeraj", "subs": 25000, "type": "small"},
        {"name": "Dubai Insider Guide", "subs": 180000, "type": "local"},
        {"name": "NY to Anywhere", "subs": 95000, "type": "small"},
    ]

    # ── Generate videos ──
    records = []
    theme_names = list(themes.keys())
    theme_weights = [themes[t]["weight"] for t in theme_names]

    # Date range: 2019-01 to 2025-12
    date_start = pd.Timestamp("2019-01-01")
    date_end = pd.Timestamp("2025-12-31")
    date_range_days = (date_end - date_start).days

    for i in range(n_videos):
        # Pick theme
        theme_key = rng.choice(theme_names, p=theme_weights)
        theme = themes[theme_key]

        # Pick title
        title = rng.choice(theme["titles"])
        # Add slight variation
        year = rng.choice(["2023", "2024", "2025", ""])
        if year and rng.random() < 0.3:
            title = title.replace("2024", year)

        # Pick channel
        channel = rng.choice(channels)

        # Views (power-law via lognormal)
        base_views = max(
            100,
            int(rng.lognormal(
                np.log(theme["avg_views"]),
                0.8,
            )),
        )
        # Boost by channel size
        channel_mult = np.clip(channel["subs"] / 1000000, 0.1, 3.0)
        views = int(base_views * channel_mult)

        # Engagement
        like_rate = np.clip(rng.normal(0.035, 0.015), 0.005, 0.10)
        comment_rate = np.clip(rng.normal(0.005, 0.003), 0.0005, 0.03)
        likes = int(views * like_rate * theme["engagement_mult"])
        comments = int(views * comment_rate * theme["engagement_mult"])

        # Duration (minutes)
        if theme_key in ["travel_vlog", "flight_review"]:
            duration_min = int(np.clip(rng.normal(18, 6), 8, 45))
        elif theme_key == "travel_guide":
            duration_min = int(np.clip(rng.normal(22, 8), 10, 60))
        else:
            duration_min = int(np.clip(rng.normal(12, 5), 4, 35))

        # Published date (weighted toward recent)
        days_ago = int(rng.exponential(365))
        pub_date = date_end - pd.Timedelta(days=min(days_ago, date_range_days))

        # Map to search query
        query_map = {
            "travel_vlog": "NYC to Dubai travel vlog",
            "travel_guide": "Dubai travel guide from New York",
            "hotel_review": "Dubai hotel review tourist",
            "things_to_do": "Dubai things to do American tourist",
            "budget_planning": "Dubai trip budget from USA",
            "flight_review": "flying to Dubai from JFK",
        }

        # Generate a description snippet
        desc_templates = [
            f"In this video I share my experience {title.lower()}. "
            f"Subscribe for more travel content!",
            f"Everything you need to know about traveling from NYC to Dubai. "
            f"Links in description!",
            f"Join me on my journey from New York to Dubai! "
            f"Don't forget to like and subscribe!",
        ]

        records.append({
            "VIDEO_ID": f"SYNTH_YT_{i:04d}",
            "TITLE": title,
            "DESCRIPTION": rng.choice(desc_templates),
            "CHANNEL_TITLE": channel["name"],
            "CHANNEL_ID": f"UC{rng.choice(list('ABCDEFGHIJKLMNOP'))}"
                          f"{i:06d}",
            "CHANNEL_SUBSCRIBERS": channel["subs"],
            "CHANNEL_TYPE": channel["type"],
            "PUBLISHED_AT": pub_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "SEARCH_QUERY": query_map.get(theme_key, "Dubai travel"),
            "CONTENT_THEME": theme_key,
            "VIEW_COUNT": views,
            "LIKE_COUNT": likes,
            "COMMENT_COUNT": comments,
            "DURATION_MINUTES": duration_min,
            "DURATION": f"PT{duration_min}M{rng.randint(0,60)}S",
        })

    df = pd.DataFrame(records)
    df["PUBLISHED_AT"] = pd.to_datetime(df["PUBLISHED_AT"])

    print(f"Generated synthetic YouTube data: {len(df)} videos")
    print(f"  Themes: {dict(df['CONTENT_THEME'].value_counts())}")
    print(f"  Total views: {df['VIEW_COUNT'].sum():,}")
    return df


# ═══════════════════════════════════════════════════════════════
# SAVE / LOAD
# ═══════════════════════════════════════════════════════════════

def save_youtube_data(df: pd.DataFrame) -> Path:
    """Save YouTube data to data/raw/youtube/"""
    output_dir = DATA_RAW / "youtube"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "youtube_videos.csv"
    df.to_csv(path, index=False)
    print(f"Saved → {path}")
    return path


def load_youtube_data() -> pd.DataFrame:
    """Load YouTube data from data/raw/youtube/"""
    path = DATA_RAW / "youtube" / "youtube_videos.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return pd.read_csv(path, parse_dates=["PUBLISHED_AT"])


# ═══════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("YouTube Data Collector")
    print("=" * 50)

    if GOOGLE_CLOUD_API_KEY and HAS_YT_CLIENT:
        print(f"API key: {GOOGLE_CLOUD_API_KEY[:8]}...\n")
        df = fetch_youtube_data()
    else:
        print("No API key or client — generating synthetic data...\n")
        df = generate_synthetic_youtube_data()

    if not df.empty:
        save_youtube_data(df)
        print(f"\n📊 Summary:")
        print(df[["CONTENT_THEME", "VIEW_COUNT", "LIKE_COUNT"]].describe())