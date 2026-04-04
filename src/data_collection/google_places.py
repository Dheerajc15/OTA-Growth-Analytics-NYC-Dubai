"""
Google Places API Client (Data Sources #1 & #2)
=================================================
Used in: M02 (Booking Funnel), M04 (Segmentation), M05 (Sentiment)

Fetches hotel/accommodation data for Dubai and NYC:
  - Text Search: find hotels by query ("luxury hotels Dubai")
  - Place Details: ratings, reviews, price_level, photos, opening_hours
  - Nearby Search: hotels within radius of coordinates
"""

import requests
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from typing import Optional

try:
    from config.settings import (
        GOOGLE_CLOUD_API_KEY, GOOGLE_PLACES_BASE_URL, DATA_RAW,
        PLACES_DUBAI_QUERIES, PLACES_NYC_QUERIES,
        DUBAI_LAT, DUBAI_LNG, NYC_LAT, NYC_LNG,
    )
except ImportError:
    GOOGLE_CLOUD_API_KEY = None
    GOOGLE_PLACES_BASE_URL = "https://maps.googleapis.com/maps/api/place"
    DATA_RAW = Path("data/raw")
    PLACES_DUBAI_QUERIES = ["luxury hotels Dubai", "budget hotels Dubai"]
    PLACES_NYC_QUERIES = ["luxury hotels Manhattan", "budget hotels NYC"]
    DUBAI_LAT, DUBAI_LNG = 25.2048, 55.2708
    NYC_LAT, NYC_LNG = 40.7128, -74.0060


# ═══════════════════════════════════════════════════════════════
# LOW-LEVEL API CALLS
# ═══════════════════════════════════════════════════════════════

def _check_api_key() -> bool:
    """Verify the Google Cloud API key is set."""
    if not GOOGLE_CLOUD_API_KEY:
        print("⚠️ GOOGLE_CLOUD_API_KEY not set in .env")
        print("   Get one at: https://console.cloud.google.com/apis/credentials")
        print("   Enable: 'Places API'")
        return False
    return True


def text_search(
    query: str,
    location: Optional[str] = None,
    radius: int = 50000,
    place_type: str = "lodging",
    max_results: int = 60,
) -> list[dict]:
    """
    Google Places Text Search — find places by text query.

    Parameters
    ----------
    query : e.g. "luxury hotels Dubai"
    location : "lat,lng" (optional — improves relevance)
    radius : search radius in meters (max 50km)
    place_type : "lodging", "restaurant", etc.
    max_results : max results across pagination (API gives 20 per page)

    Returns list of raw Place dicts.
    """
    if not _check_api_key():
        return []

    url = f"{GOOGLE_PLACES_BASE_URL}/textsearch/json"
    params = {
        "query": query,
        "type": place_type,
        "key": GOOGLE_CLOUD_API_KEY,
    }
    if location:
        params["location"] = location
        params["radius"] = radius

    all_results = []
    page = 1

    while True:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()

        status = data.get("status", "UNKNOWN")
        if status != "OK":
            if status == "ZERO_RESULTS":
                print(f"    No results for: '{query}'")
            else:
                print(f"    API error ({status}): {data.get('error_message', '')}")
            break

        results = data.get("results", [])
        all_results.extend(results)
        print(f"    Page {page}: +{len(results)} places (total: {len(all_results)})")

        next_token = data.get("next_page_token")
        if not next_token or len(all_results) >= max_results:
            break

        time.sleep(2.5)
        params = {"pagetoken": next_token, "key": GOOGLE_CLOUD_API_KEY}
        page += 1

    return all_results[:max_results]


def get_place_details(place_id: str) -> Optional[dict]:
    """
    Fetch detailed info for a single place (reviews, price_level, etc.)
    """
    if not _check_api_key():
        return None

    url = f"{GOOGLE_PLACES_BASE_URL}/details/json"
    fields = (
        "name,rating,user_ratings_total,price_level,formatted_address,"
        "geometry,types,reviews,website,url,photos,business_status,"
        "opening_hours,formatted_phone_number"
    )
    params = {
        "place_id": place_id,
        "fields": fields,
        "key": GOOGLE_CLOUD_API_KEY,
    }

    resp = requests.get(url, params=params, timeout=30)
    data = resp.json()

    if data.get("status") != "OK":
        return None

    return data.get("result")


# ═══════════════════════════════════════════════════════════════
# HIGH-LEVEL: FETCH HOTEL DATA FOR A MARKET
# ═══════════════════════════════════════════════════════════════

def fetch_hotels_for_market(
    queries: list[str],
    market_name: str,
    location: Optional[str] = None,
    fetch_details: bool = True,
    max_per_query: int = 60,
    detail_delay: float = 0.3,
) -> pd.DataFrame:
    """
    Fetch hotel data for an entire market (Dubai or NYC).
    """
    print(f"\n{'='*60}")
    print(f"  Fetching {market_name} Hotels")
    print(f"  Queries: {len(queries)} | Details: {fetch_details}")
    print(f"{'='*60}")

    all_places = {}

    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] '{query}'")
        results = text_search(query, location=location, max_results=max_per_query)

        for place in results:
            pid = place.get("place_id")
            if pid and pid not in all_places:
                all_places[pid] = place

        time.sleep(1)

    print(f"\n📍 Unique hotels found: {len(all_places)}")

    if not all_places:
        return pd.DataFrame()

    records = []
    for pid, place in all_places.items():
        geo = place.get("geometry", {}).get("location", {})
        record = {
            "PLACE_ID": pid,
            "NAME": place.get("name", ""),
            "MARKET": market_name,
            "RATING": place.get("rating"),
            "TOTAL_RATINGS": place.get("user_ratings_total", 0),
            "PRICE_LEVEL": place.get("price_level"),
            "ADDRESS": place.get("formatted_address", ""),
            "LAT": geo.get("lat"),
            "LNG": geo.get("lng"),
            "BUSINESS_STATUS": place.get("business_status", ""),
            "TYPES": ", ".join(place.get("types", [])),
        }
        records.append(record)

    if fetch_details and _check_api_key():
        print(f"\n📋 Fetching details for {len(records)} hotels...")
        for i, record in enumerate(records):
            details = get_place_details(record["PLACE_ID"])
            if details:
                record["WEBSITE"] = details.get("website", "")
                record["GOOGLE_URL"] = details.get("url", "")
                record["PHONE"] = details.get("formatted_phone_number", "")
                record["NUM_PHOTOS"] = len(details.get("photos", []))

                reviews = details.get("reviews", [])
                record["NUM_REVIEWS_FETCHED"] = len(reviews)
                if reviews:
                    review_ratings = [r.get("rating", 0) for r in reviews]
                    record["AVG_REVIEW_RATING"] = round(np.mean(review_ratings), 2)
                    record["REVIEW_TEXTS"] = " ||| ".join(
                        [r.get("text", "") for r in reviews if r.get("text")]
                    )
                else:
                    record["AVG_REVIEW_RATING"] = None
                    record["REVIEW_TEXTS"] = ""

                oh = details.get("opening_hours", {})
                record["OPEN_NOW"] = oh.get("open_now")

            if (i + 1) % 25 == 0:
                print(f"    Details fetched: {i+1}/{len(records)}")
            time.sleep(detail_delay)

        print(f"    ✅ Details complete for {len(records)} hotels")

    df = pd.DataFrame(records)
    print(f"\n✅ {market_name}: {len(df)} hotels loaded")
    return df


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE: FETCH BOTH MARKETS
# ═══════════════════════════════════════════════════════════════

def fetch_dubai_hotels(fetch_details: bool = True) -> pd.DataFrame:
    """Fetch all Dubai hotel data."""
    return fetch_hotels_for_market(
        queries=PLACES_DUBAI_QUERIES,
        market_name="Dubai",
        location=f"{DUBAI_LAT},{DUBAI_LNG}",
        fetch_details=fetch_details,
    )


def fetch_nyc_hotels(fetch_details: bool = True) -> pd.DataFrame:
    """Fetch all NYC hotel data."""
    return fetch_hotels_for_market(
        queries=PLACES_NYC_QUERIES,
        market_name="NYC",
        location=f"{NYC_LAT},{NYC_LNG}",
        fetch_details=fetch_details,
    )


def fetch_both_markets(fetch_details: bool = True) -> pd.DataFrame:
    """Fetch Dubai + NYC and combine into single DataFrame."""
    dubai = fetch_dubai_hotels(fetch_details=fetch_details)
    nyc = fetch_nyc_hotels(fetch_details=fetch_details)
    combined = pd.concat([dubai, nyc], ignore_index=True)
    print(f"\n📊 Combined: {len(combined)} hotels ({len(dubai)} Dubai + {len(nyc)} NYC)")
    return combined


# ═══════════════════════════════════════════════════════════════
# SAVE / LOAD
# ═══════════════════════════════════════════════════════════════

def save_places_data(df: pd.DataFrame, market: str) -> Path:
    """Save to data/raw/google_places_{market}/"""
    market_lower = market.lower().replace(" ", "_")
    output_dir = DATA_RAW / f"google_places_{market_lower}"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{market_lower}_hotels.csv"
    df.to_csv(path, index=False)
    print(f"Saved → {path}")
    return path


def load_places_data(market: str) -> pd.DataFrame:
    """Load from data/raw/google_places_{market}/"""
    market_lower = market.lower().replace(" ", "_")
    path = DATA_RAW / f"google_places_{market_lower}" / f"{market_lower}_hotels.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return pd.read_csv(path)


# ═══════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Google Places API — Hotel Data Collector")
    print("=" * 50)

    if GOOGLE_CLOUD_API_KEY:
        print(f"API key: {GOOGLE_CLOUD_API_KEY[:8]}...\n")
        combined = fetch_both_markets(fetch_details=True)
        if not combined.empty:
            for market in ["Dubai", "NYC"]:
                mdf = combined[combined["MARKET"] == market]
                save_places_data(mdf, market)
            print(f"\nSummary:")
            print(combined.groupby("MARKET")[["RATING", "TOTAL_RATINGS", "PRICE_LEVEL"]].describe())
    else:
        print("No API key. Use: python scripts/generate_seeds.py")
