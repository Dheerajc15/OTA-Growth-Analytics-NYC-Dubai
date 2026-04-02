"""
Google Places API Client (Data Sources #1 & #2)
=================================================
Used in: M02 (Booking Funnel), M04 (Segmentation), M05 (Sentiment)

Fetches hotel/accommodation data for Dubai and NYC:
  - Text Search: find hotels by query ("luxury hotels Dubai")
  - Place Details: ratings, reviews, price_level, photos, opening_hours
  - Nearby Search: hotels within radius of coordinates

API Reference: https://developers.google.com/maps/documentation/places/web-service
Free tier: $200/month credit ≈ 10,000 detail requests
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
# ════════════════════════════════════════════════════��══════════

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

        # Check for next page
        next_token = data.get("next_page_token")
        if not next_token or len(all_results) >= max_results:
            break

        # Google requires ~2s delay before next_page_token is valid
        time.sleep(2.5)
        params = {"pagetoken": next_token, "key": GOOGLE_CLOUD_API_KEY}
        page += 1

    return all_results[:max_results]


def get_place_details(place_id: str) -> Optional[dict]:
    """
    Fetch detailed info for a single place (reviews, price_level, etc.)

    Fields requested: name, rating, user_ratings_total, price_level,
    formatted_address, geometry, types, reviews, website, url, photos,
    business_status, opening_hours
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

    Pipeline:
      1. Text Search for each query → get place_ids
      2. Deduplicate by place_id
      3. (Optional) Fetch Place Details for each unique hotel
      4. Return structured DataFrame

    Parameters
    ----------
    queries : list of search queries
    market_name : "Dubai" or "NYC" — added as MARKET column
    location : "lat,lng" to bias results
    fetch_details : whether to call Place Details API (costs more quota)
    max_per_query : max results per text search query
    detail_delay : seconds between detail calls (rate limiting)

    Returns
    -------
    pd.DataFrame with hotel-level data
    """
    print(f"\n{'='*60}")
    print(f"  Fetching {market_name} Hotels")
    print(f"  Queries: {len(queries)} | Details: {fetch_details}")
    print(f"{'='*60}")

    # Step 1: Text Search across all queries
    all_places = {}  # place_id → place dict (dedup)

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

    # Step 2: Parse text search results into records
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

    # Step 3: Optionally fetch detailed info (reviews, website, etc.)
    if fetch_details and _check_api_key():
        print(f"\n📋 Fetching details for {len(records)} hotels...")
        for i, record in enumerate(records):
            details = get_place_details(record["PLACE_ID"])
            if details:
                record["WEBSITE"] = details.get("website", "")
                record["GOOGLE_URL"] = details.get("url", "")
                record["PHONE"] = details.get("formatted_phone_number", "")
                record["NUM_PHOTOS"] = len(details.get("photos", []))

                # Extract reviews
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

                # Opening hours
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
# SYNTHETIC DATA (for development without API key / quota)
# ═══════════════════════════════════════════════════════════════

def generate_synthetic_hotels(seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic hotel data for Dubai and NYC.

    Simulates real market characteristics:
    - Dubai: higher prices, fewer listings, luxury-skewed, higher avg ratings
    - NYC: lower prices, massive listing density, wider spread, more reviews

    FIXES:
    - Bug 1: BUSINESS_STATUS uses proper 2-value choice
    - Bug 2: AVG_REVIEW_RATING uses explicit conditional (no lambda ambiguity)
    - Bug 3: NYC hotels now have realistic website URLs
    """
    rng = np.random.RandomState(seed)

    # ── Dubai Hotels ──
    n_dubai = 180
    dubai_neighborhoods = [
        "Dubai Marina", "Downtown Dubai", "Palm Jumeirah", "Deira",
        "JBR", "Business Bay", "Jumeirah", "Al Barsha", "Dubai Creek",
        "DIFC", "Festival City", "Sheikh Zayed Road",
    ]
    dubai_prefixes = [
        "Atlantis", "Burj", "Jumeirah", "Palazzo", "Ritz-Carlton",
        "Marriott", "Hilton", "Sofitel", "Rotana", "Address",
        "Anantara", "One&Only", "Taj", "Oberoi", "Le Meridien",
        "Grand Hyatt", "Kempinski", "Shangri-La", "Four Points",
        "W Hotel", "St. Regis", "JW Marriott", "Conrad", "Vida",
    ]

    dubai_records = []
    for i in range(n_dubai):
        neighborhood = rng.choice(dubai_neighborhoods)
        name = f"{rng.choice(dubai_prefixes)} {neighborhood}"

        # Dubai is luxury-skewed: price_level 3-4 dominant
        price_level = rng.choice([1, 2, 3, 4], p=[0.05, 0.20, 0.40, 0.35])
        # Ratings: higher than NYC avg (luxury = better service)
        rating = np.clip(rng.normal(4.3, 0.4), 2.5, 5.0)
        total_ratings = int(rng.lognormal(7.0, 1.2))  # fewer reviews than NYC

        # FIX Bug 1: Clean 2-value choice for BUSINESS_STATUS
        business_status = rng.choice(
            ["OPERATIONAL", "CLOSED_TEMPORARILY"],
            p=[0.98, 0.02],
        )

        dubai_records.append({
            "PLACE_ID": f"SYNTH_DUBAI_{i:04d}",
            "NAME": name,
            "MARKET": "Dubai",
            "RATING": round(rating, 1),
            "TOTAL_RATINGS": total_ratings,
            "PRICE_LEVEL": price_level,
            "ADDRESS": f"{neighborhood}, Dubai, UAE",
            "LAT": DUBAI_LAT + rng.uniform(-0.08, 0.08),
            "LNG": DUBAI_LNG + rng.uniform(-0.08, 0.08),
            "BUSINESS_STATUS": business_status,
            "TYPES": "lodging, point_of_interest, establishment",
            "NEIGHBORHOOD": neighborhood,
            "NUM_PHOTOS": rng.randint(3, 30),
            "NUM_REVIEWS_FETCHED": min(5, max(0, int(rng.normal(4, 1.5)))),
            "WEBSITE": f"https://www.{name.lower().replace(' ', '').replace('&', '')}.com",
        })

    # ── NYC Hotels ──
    n_nyc = 280
    nyc_neighborhoods = [
        "Times Square", "Midtown East", "Midtown West", "Chelsea",
        "SoHo", "Lower Manhattan", "Upper East Side", "Upper West Side",
        "Brooklyn Heights", "Williamsburg", "Long Island City",
        "Greenwich Village", "East Village", "Hell's Kitchen",
    ]
    nyc_prefixes = [
        "The Standard", "The Plaza", "Hyatt", "Marriott", "Hilton",
        "Pod", "Moxy", "EVEN", "Arlo", "Ace", "Hotel Indigo",
        "citizenM", "Hampton Inn", "Holiday Inn", "Sheraton",
        "The Westin", "Courtyard", "SpringHill", "Fairfield",
        "Best Western", "Comfort Inn", "La Quinta", "Doubletree",
    ]

    nyc_records = []
    for i in range(n_nyc):
        neighborhood = rng.choice(nyc_neighborhoods)
        name = f"{rng.choice(nyc_prefixes)} {neighborhood}"

        # NYC: wider price spread, more budget options
        price_level = rng.choice([1, 2, 3, 4], p=[0.15, 0.40, 0.30, 0.15])
        rating = np.clip(rng.normal(4.0, 0.5), 2.0, 5.0)
        total_ratings = int(rng.lognormal(7.5, 1.3)) 

        clean_name = name.lower().replace(" ", "").replace("'", "")
        website = f"https://www.{clean_name}.com" if rng.random() < 0.75 else ""

        nyc_records.append({
            "PLACE_ID": f"SYNTH_NYC_{i:04d}",
            "NAME": name,
            "MARKET": "NYC",
            "RATING": round(rating, 1),
            "TOTAL_RATINGS": total_ratings,
            "PRICE_LEVEL": price_level,
            "ADDRESS": f"{neighborhood}, New York, NY, USA",
            "LAT": NYC_LAT + rng.uniform(-0.06, 0.06),
            "LNG": NYC_LNG + rng.uniform(-0.06, 0.06),
            "BUSINESS_STATUS": "OPERATIONAL",
            "TYPES": "lodging, point_of_interest, establishment",
            "NEIGHBORHOOD": neighborhood,
            "NUM_PHOTOS": rng.randint(2, 25),
            "NUM_REVIEWS_FETCHED": min(5, max(0, int(rng.normal(4, 1)))),
            "WEBSITE": website,
        })

    # ── Combine ──
    dubai_df = pd.DataFrame(dubai_records)
    nyc_df = pd.DataFrame(nyc_records)
    combined = pd.concat([dubai_df, nyc_df], ignore_index=True)

    # Generate synthetic review text snippets
    dubai_sentiments = [
        "Absolutely stunning views of the Burj Khalifa from our room.",
        "Service was impeccable, truly 5-star experience.",
        "Overpriced for what you get. Beach was nice though.",
        "Perfect for business travel, very close to DIFC.",
        "The pool area was amazing but the room was small for the price.",
        "Best hotel I've ever stayed in. The breakfast buffet was insane.",
        "Location is great but construction noise was terrible.",
        "Amazing desert safari arranged by concierge.",
        "AC worked perfectly, which is essential in Dubai heat.",
        "Felt like a resort, not just a hotel. Would come back.",
    ]
    nyc_sentiments = [
        "Tiny room but great location near Times Square.",
        "Perfect for a quick business trip. Walkable to everything.",
        "Way too expensive for such a small room. Classic NYC.",
        "Staff was friendly and helpful with restaurant recommendations.",
        "Noisy street but good soundproofing in the room.",
        "Great rooftop bar with Manhattan skyline views.",
        "Clean and modern, good value for Midtown.",
        "Bed was uncomfortable but location can't be beat.",
        "Loved the subway access, made exploring easy.",
        "Boutique hotel with real character. Not cookie-cutter.",
    ]

    def _gen_reviews(market, n):
        pool = dubai_sentiments if market == "Dubai" else nyc_sentiments
        reviews = [rng.choice(pool) for _ in range(n)]
        return " ||| ".join(reviews)

    combined["REVIEW_TEXTS"] = combined.apply(
        lambda r: _gen_reviews(r["MARKET"], r["NUM_REVIEWS_FETCHED"]), axis=1
    )

    avg_ratings = []
    for _, row in combined.iterrows():
        if row["NUM_REVIEWS_FETCHED"] > 0:
            val = round(np.clip(row["RATING"] + rng.uniform(-0.3, 0.3), 1, 5), 2)
        else:
            val = None
        avg_ratings.append(val)
    combined["AVG_REVIEW_RATING"] = avg_ratings

    print(f"Generated synthetic hotel data: {len(combined)} hotels "
          f"({n_dubai} Dubai + {n_nyc} NYC)")
    return combined


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
    else:
        print("No API key — generating synthetic data...\n")
        combined = generate_synthetic_hotels()
        for market in ["Dubai", "NYC"]:
            mdf = combined[combined["MARKET"] == market]
            save_places_data(mdf, market)

    print(f"\n📊 Summary:")
    print(combined.groupby("MARKET")[["RATING", "TOTAL_RATINGS", "PRICE_LEVEL"]].describe())