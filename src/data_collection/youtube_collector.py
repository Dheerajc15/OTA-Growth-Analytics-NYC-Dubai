"""
YouTube Data API v3 Collector 
================================================
Fetches YouTube video metadata for NYC->Dubai travel content:
  - Search by queries ("NYC to Dubai travel vlog", etc.)
  - Video statistics: views, likes, comments
  - Channel metadata
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
        YOUTUBE_MAX_RESULTS_PER_QUERY, DATA_RAW,
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
        print("GOOGLE_CLOUD_API_KEY not set — cannot use YouTube API")
        return None
    if not HAS_YT_CLIENT:
        print("google-api-python-client not installed")
        return None
    return yt_build("youtube", "v3", developerKey=GOOGLE_CLOUD_API_KEY)


def search_youtube_videos(
    query: str,
    max_results: int = 50,
    order: str = "relevance",
    published_after: Optional[str] = None,
) -> list[dict]:
    """Search YouTube for videos matching a query."""
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
    """Fetch view/like/comment counts for a batch of video IDs."""
    youtube = _get_youtube_service()
    if not youtube:
        return {}

    stats = {}
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
    """Full pipeline: search all queries -> fetch stats -> return DataFrame."""
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
    print(f"  {len(df)} videos with statistics")
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
    print(f"Saved -> {path}")
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
        if not df.empty:
            save_youtube_data(df)
            print(f"\nSummary:")
            print(df[["CONTENT_THEME", "VIEW_COUNT", "LIKE_COUNT"]].describe())
    else:
        print("No API key or client. Run: python scripts/generate_seeds.py")
