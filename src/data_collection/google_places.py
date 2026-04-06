from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

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


def _check_api_key() -> bool:
    if not GOOGLE_CLOUD_API_KEY:
        print("⚠️ GOOGLE_CLOUD_API_KEY not set in .env")
        return False
    return True


def text_search(
    query: str,
    location: Optional[str] = None,
    radius: int = 50000,
    place_type: str = "lodging",
    max_results: int = 60,
) -> list[dict]:
    if not _check_api_key():
        return []

    url = f"{GOOGLE_PLACES_BASE_URL}/textsearch/json"
    params = {"query": query, "type": place_type, "key": GOOGLE_CLOUD_API_KEY}
    if location:
        params["location"] = location
        params["radius"] = radius

    all_results = []
    page = 1

    while True:
        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
        except Exception as e:
            print(f"    Request error: {e}")
            break

        status = data.get("status", "UNKNOWN")
        if status != "OK":
            if status != "ZERO_RESULTS":
                print(f"    API error ({status}): {data.get('error_message', '')}")
            break

        results = data.get("results", [])
        all_results.extend(results)
        print(f"    Page {page}: +{len(results)} (total: {len(all_results)})")

        next_token = data.get("next_page_token")
        if not next_token or len(all_results) >= max_results:
            break

        time.sleep(2.5)  # token warmup
        params = {"pagetoken": next_token, "key": GOOGLE_CLOUD_API_KEY}
        page += 1

    return all_results[:max_results]


def get_place_details(place_id: str) -> Optional[dict]:
    if not _check_api_key():
        return None

    url = f"{GOOGLE_PLACES_BASE_URL}/details/json"
    fields = (
        "name,rating,user_ratings_total,price_level,formatted_address,"
        "geometry,types,reviews,website,url,photos,business_status,"
        "opening_hours,formatted_phone_number"
    )
    params = {"place_id": place_id, "fields": fields, "key": GOOGLE_CLOUD_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
    except Exception:
        return None

    if data.get("status") != "OK":
        return None
    return data.get("result")


def fetch_hotels_for_market(
    queries: list[str],
    market_name: str,
    location: Optional[str] = None,
    fetch_details: bool = True,
    max_per_query: int = 60,
    detail_delay: float = 0.3,
) -> pd.DataFrame:
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

    if not all_places:
        return pd.DataFrame()

    records = []
    for pid, place in all_places.items():
        geo = place.get("geometry", {}).get("location", {})
        records.append({
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
            "WEBSITE": "",
            "GOOGLE_URL": "",
            "PHONE": "",
            "NUM_PHOTOS": 0,
            "NUM_REVIEWS_FETCHED": 0,
            "AVG_REVIEW_RATING": None,
            "REVIEW_TEXTS": "",
            "OPEN_NOW": None,
        })

    if fetch_details and _check_api_key():
        print(f"\n📋 Fetching details for {len(records)} hotels...")
        for i, rec in enumerate(records):
            details = get_place_details(rec["PLACE_ID"])
            if details:
                rec["WEBSITE"] = details.get("website", "")
                rec["GOOGLE_URL"] = details.get("url", "")
                rec["PHONE"] = details.get("formatted_phone_number", "")
                rec["NUM_PHOTOS"] = len(details.get("photos", []))

                reviews = details.get("reviews", [])
                rec["NUM_REVIEWS_FETCHED"] = len(reviews)
                if reviews:
                    rr = [r.get("rating", 0) for r in reviews]
                    rec["AVG_REVIEW_RATING"] = round(float(np.mean(rr)), 2)
                    rec["REVIEW_TEXTS"] = " ||| ".join([r.get("text", "") for r in reviews if r.get("text")])

                rec["OPEN_NOW"] = details.get("opening_hours", {}).get("open_now")

            if (i + 1) % 25 == 0:
                print(f"    Details fetched: {i+1}/{len(records)}")
            time.sleep(detail_delay)

    df = pd.DataFrame(records)
    print(f"\n✅ {market_name}: {len(df)} hotels loaded")
    return df


def fetch_dubai_hotels(fetch_details: bool = True) -> pd.DataFrame:
    return fetch_hotels_for_market(
        queries=PLACES_DUBAI_QUERIES,
        market_name="Dubai",
        location=f"{DUBAI_LAT},{DUBAI_LNG}",
        fetch_details=fetch_details,
    )


def fetch_nyc_hotels(fetch_details: bool = True) -> pd.DataFrame:
    return fetch_hotels_for_market(
        queries=PLACES_NYC_QUERIES,
        market_name="NYC",
        location=f"{NYC_LAT},{NYC_LNG}",
        fetch_details=fetch_details,
    )


def fetch_both_markets(fetch_details: bool = True) -> pd.DataFrame:
    dubai = fetch_dubai_hotels(fetch_details=fetch_details)
    nyc = fetch_nyc_hotels(fetch_details=fetch_details)
    return pd.concat([dubai, nyc], ignore_index=True)


def save_places_data(df: pd.DataFrame, market: str) -> Path:
    market_lower = market.lower().replace(" ", "_")
    out_dir = DATA_RAW / f"google_places_{market_lower}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{market_lower}_hotels.csv"
    df.to_csv(path, index=False)
    print(f"Saved → {path}")
    return path


def load_places_data(market: str) -> pd.DataFrame:
    market_lower = market.lower().replace(" ", "_")
    path = DATA_RAW / f"google_places_{market_lower}" / f"{market_lower}_hotels.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return pd.read_csv(path)


# Backward-compatible alias
fetch_google_places_data = fetch_both_markets