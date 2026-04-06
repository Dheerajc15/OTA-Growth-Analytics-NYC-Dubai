from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from config.settings import (
        GOOGLE_CLOUD_API_KEY, YOUTUBE_SEARCH_QUERIES,
        YOUTUBE_MAX_RESULTS_PER_QUERY, DATA_RAW,
    )
except ImportError:
    GOOGLE_CLOUD_API_KEY = None
    YOUTUBE_SEARCH_QUERIES = []
    YOUTUBE_MAX_RESULTS_PER_QUERY = 50
    DATA_RAW = Path("data/raw")

try:
    from googleapiclient.discovery import build as yt_build
    HAS_YT_CLIENT = True
except ImportError:
    HAS_YT_CLIENT = False


def _service():
    if not GOOGLE_CLOUD_API_KEY or not HAS_YT_CLIENT:
        return None
    return yt_build("youtube", "v3", developerKey=GOOGLE_CLOUD_API_KEY)


def search_youtube_videos(query: str, max_results: int = 50, order: str = "relevance") -> list[dict]:
    yt = _service()
    if yt is None:
        return []
    try:
        resp = yt.search().list(
            q=query, part="snippet", type="video", maxResults=min(max_results, 50), order=order
        ).execute()
    except Exception:
        return []

    out = []
    for it in resp.get("items", []):
        s = it.get("snippet", {})
        out.append({
            "VIDEO_ID": it["id"]["videoId"],
            "TITLE": s.get("title", ""),
            "DESCRIPTION": s.get("description", ""),
            "CHANNEL_TITLE": s.get("channelTitle", ""),
            "CHANNEL_ID": s.get("channelId", ""),
            "PUBLISHED_AT": s.get("publishedAt", ""),
            "SEARCH_QUERY": query,
        })
    return out


def get_video_statistics(video_ids: list[str]) -> dict:
    yt = _service()
    if yt is None:
        return {}
    stats = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        try:
            resp = yt.videos().list(part="statistics,contentDetails", id=",".join(batch)).execute()
        except Exception:
            continue
        for it in resp.get("items", []):
            s = it.get("statistics", {})
            cd = it.get("contentDetails", {})
            stats[it["id"]] = {
                "VIEW_COUNT": int(s.get("viewCount", 0)),
                "LIKE_COUNT": int(s.get("likeCount", 0)),
                "COMMENT_COUNT": int(s.get("commentCount", 0)),
                "DURATION": cd.get("duration", ""),
            }
        time.sleep(0.2)
    return stats


def fetch_youtube_data(queries: list[str] | None = None, max_per_query: int | None = None) -> pd.DataFrame:
    queries = queries or YOUTUBE_SEARCH_QUERIES
    max_per_query = max_per_query or YOUTUBE_MAX_RESULTS_PER_QUERY

    videos = {}
    for q in queries:
        for v in search_youtube_videos(q, max_per_query):
            videos.setdefault(v["VIDEO_ID"], v)

    if not videos:
        return pd.DataFrame()

    stats = get_video_statistics(list(videos.keys()))
    for vid, s in stats.items():
        videos[vid].update(s)

    return pd.DataFrame(videos.values())


def save_youtube_data(df: pd.DataFrame) -> Path:
    out = DATA_RAW / "youtube"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "youtube_videos.csv"
    df.to_csv(path, index=False)
    return path


def load_youtube_data() -> pd.DataFrame:
    path = DATA_RAW / "youtube" / "youtube_videos.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=["PUBLISHED_AT"])