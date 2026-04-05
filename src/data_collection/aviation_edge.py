"""
Aviation Edge API Client 
==========================================

Provides:
  - Flight routes: which airlines fly JFK/EWR -> DXB, frequency
  - Flight schedules/timetables: departure times, days of week
  - Airline database: carrier details
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from typing import Optional

try:
    from config.settings import (
        AVIATION_EDGE_API_KEY, AVIATION_EDGE_BASE_URL,
        AVIATION_EDGE_ENDPOINTS, DATA_RAW,
        ORIGIN_AIRPORTS, DESTINATION_AIRPORT,
    )
except ImportError:
    AVIATION_EDGE_API_KEY = None
    AVIATION_EDGE_BASE_URL = "https://aviation-edge.com/v2/public"
    AVIATION_EDGE_ENDPOINTS = {
        "routes": "/routes",
        "timetable": "/timetable",
        "airlines": "/airlineDatabase",
        "airports": "/airportDatabase",
    }
    DATA_RAW = Path("data/raw")
    ORIGIN_AIRPORTS = ["JFK", "EWR", "LGA"]
    DESTINATION_AIRPORT = "DXB"


# ═══════════════════════════════════════════════════════════════
# API HELPERS
# ═══════════════════════════════════════════════════════════════

def _make_request(endpoint: str, params: dict) -> Optional[list]:
    """Make authenticated request to Aviation Edge API."""
    if not AVIATION_EDGE_API_KEY:
        print("AVIATION_EDGE_API_KEY not set in .env")
        return None

    url = f"{AVIATION_EDGE_BASE_URL}{endpoint}"
    params["key"] = AVIATION_EDGE_API_KEY

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and "error" in data:
            print(f"API error: {data['error']}")
            return None

        return data if isinstance(data, list) else [data]
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# ROUTE DATA — Which airlines fly NYC -> DXB
# ═══════════════════════════════════════════════════════════════

def fetch_routes(
    origins: Optional[list[str]] = None,
    destination: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch airline routes from NYC airports to Dubai."""
    origins = origins or ORIGIN_AIRPORTS
    destination = destination or DESTINATION_AIRPORT

    all_routes = []
    for origin in origins:
        print(f"  Fetching routes: {origin} -> {destination}")
        data = _make_request(
            AVIATION_EDGE_ENDPOINTS["routes"],
            {"departureIata": origin, "arrivalIata": destination},
        )
        if data:
            for route in data:
                route["queried_origin"] = origin
            all_routes.extend(data)
            print(f"    Got {len(data)} route(s)")
        else:
            print(f"    No data / API error")
        time.sleep(1)

    if not all_routes:
        return pd.DataFrame()

    df = pd.DataFrame(all_routes)
    print(f"Total routes: {len(df)}")
    return df


# ═══════════════════════════════════════════════════════════════
# TIMETABLE — Flight schedules
# ═══════════════════════════════════════════════════════════════

def fetch_timetable(
    origin: str = "JFK",
    destination: str = None,
    timetable_type: str = "departure",
) -> pd.DataFrame:
    """Fetch flight timetable (schedules) from an airport."""
    destination = destination or DESTINATION_AIRPORT

    print(f"Fetching timetable: {origin} departures to {destination}")
    data = _make_request(
        AVIATION_EDGE_ENDPOINTS["timetable"],
        {"iataCode": origin, "type": timetable_type},
    )

    if not data:
        return pd.DataFrame()

    records = []
    for flight in data:
        try:
            record = {
                "flight_iata": flight.get("flight", {}).get("iataNumber", ""),
                "airline_iata": flight.get("airline", {}).get("iataCode", ""),
                "airline_name": flight.get("airline", {}).get("name", ""),
                "departure_airport": flight.get("departure", {}).get("iataCode", ""),
                "departure_scheduled": flight.get("departure", {}).get("scheduledTime", ""),
                "departure_terminal": flight.get("departure", {}).get("terminal", ""),
                "arrival_airport": flight.get("arrival", {}).get("iataCode", ""),
                "arrival_scheduled": flight.get("arrival", {}).get("scheduledTime", ""),
                "arrival_terminal": flight.get("arrival", {}).get("terminal", ""),
                "status": flight.get("status", ""),
                "aircraft_iata": flight.get("aircraft", {}).get("iataCode", ""),
            }
            records.append(record)
        except (KeyError, TypeError):
            continue

    df = pd.DataFrame(records)

    if destination and "arrival_airport" in df.columns:
        df = df[df["arrival_airport"] == destination]

    print(f"Timetable: {len(df)} flights ({origin} -> {destination})")
    return df


# ═══════════════════════════════════════════════════════════════
# AGGREGATE: ALL NYC -> DXB FLIGHT DATA
# ═══════════════════════════════════════════════════════════════

def fetch_all_nyc_dubai_flights() -> pd.DataFrame:
    """Fetch timetables from all NYC airports to DXB and combine."""
    all_flights = []
    for origin in ORIGIN_AIRPORTS:
        flights = fetch_timetable(origin=origin)
        if not flights.empty:
            all_flights.append(flights)
        time.sleep(2)

    if not all_flights:
        print("No flight data retrieved for any NYC airport.")
        return pd.DataFrame()

    combined = pd.concat(all_flights, ignore_index=True)
    print(f"Combined: {len(combined)} flights across {len(all_flights)} airports")
    return combined


def compute_flight_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily/weekly flight frequency by airline for demand analysis."""
    if df.empty:
        return df

    freq = (
        df.groupby(["departure_airport", "airline_iata", "airline_name"])
        .agg(
            num_flights=("flight_iata", "count"),
            unique_flights=("flight_iata", "nunique"),
        )
        .reset_index()
        .sort_values("num_flights", ascending=False)
    )

    freq["market_share_pct"] = (
        freq["num_flights"] / freq["num_flights"].sum() * 100
    ).round(2)

    return freq


# ═══════════════════════════════════════════════════════════════
# SAVE / LOAD
# ═══════════════════════════════════════════════════════════════

def save_aviation_data(df: pd.DataFrame, filename: str) -> Path:
    output_dir = DATA_RAW / "aviation_edge"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.to_csv(path, index=False)
    print(f"Saved -> {path}")
    return path


def load_aviation_data(filename: str) -> pd.DataFrame:
    path = DATA_RAW / "aviation_edge" / filename
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return pd.read_csv(path)


# ═══════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Aviation Edge API — Test")
    print("=" * 40)

    if AVIATION_EDGE_API_KEY:
        print("API key found — fetching live data...\n")
        routes = fetch_routes()
        if not routes.empty:
            save_aviation_data(routes, "routes_nyc_dxb.csv")

        flights = fetch_all_nyc_dubai_flights()
        if not flights.empty:
            save_aviation_data(flights, "flights_nyc_dxb.csv")
            freq = compute_flight_frequency(flights)
            print("\nAirline Frequency:")
            print(freq.to_string(index=False))
    else:
        print("No API key. Run: python scripts/generate_seeds.py")
