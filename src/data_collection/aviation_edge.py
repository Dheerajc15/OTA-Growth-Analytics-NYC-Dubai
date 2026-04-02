"""
Aviation Edge API Client (Data Source #4)
==========================================
Used in: M01 (Demand Forecasting), M06 (Visa & Friction Analysis)

Provides:
  - Flight routes: which airlines fly JFK/EWR → DXB, frequency
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
        print("⚠️ AVIATION_EDGE_API_KEY not set in .env")
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
# ROUTE DATA — Which airlines fly NYC → DXB
# ═══════════════════════════════════════════════════════════════

def fetch_routes(
    origins: Optional[list[str]] = None,
    destination: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch airline routes from NYC airports to Dubai.

    Returns DataFrame with: airline, origin, destination, flight numbers.
    """
    origins = origins or ORIGIN_AIRPORTS
    destination = destination or DESTINATION_AIRPORT

    all_routes = []
    for origin in origins:
        print(f"  Fetching routes: {origin} → {destination}")
        data = _make_request(
            AVIATION_EDGE_ENDPOINTS["routes"],
            {"departureIata": origin, "arrivalIata": destination},
        )
        if data:
            for route in data:
                route["queried_origin"] = origin
            all_routes.extend(data)
            print(f"    ✓ {len(data)} route(s)")
        else:
            print(f"    ✗ No data / API error")
        time.sleep(1)

    if not all_routes:
        return pd.DataFrame()

    df = pd.DataFrame(all_routes)
    print(f"\n✅ Total routes: {len(df)}")
    return df


# ═══════════════════════════════════════════════════════════════
# TIMETABLE — Flight schedules
# ═══════════════════════════════════════════════════════════════

def fetch_timetable(
    origin: str = "JFK",
    destination: str = None,
    timetable_type: str = "departure",
) -> pd.DataFrame:
    """
    Fetch flight timetable (schedules) from an airport.

    Parameters
    ----------
    origin : IATA airport code
    destination : filter to specific destination (optional)
    timetable_type : "departure" or "arrival"
    """
    destination = destination or DESTINATION_AIRPORT

    print(f"Fetching timetable: {origin} departures to {destination}")
    data = _make_request(
        AVIATION_EDGE_ENDPOINTS["timetable"],
        {"iataCode": origin, "type": timetable_type},
    )

    if not data:
        return pd.DataFrame()

    # The timetable response is nested — flatten key fields
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

    # Filter to DXB destination
    if destination and "arrival_airport" in df.columns:
        df = df[df["arrival_airport"] == destination]

    print(f"✅ Timetable: {len(df)} flights ({origin} → {destination})")
    return df


# ═══════════════════════════════════════════════════════════════
# AGGREGATE: ALL NYC → DXB FLIGHT DATA
# ═══════════════════════════════════════════════════════════════

def fetch_all_nyc_dubai_flights() -> pd.DataFrame:
    """Fetch timetables from all NYC airports to DXB and combine."""
    all_flights = []
    for origin in ORIGIN_AIRPORTS:
        flights = fetch_timetable(origin=origin)
        if not flights.empty:
            all_flights.append(flights)
        time.sleep(2)  # rate-limit

    if not all_flights:
        print("⚠️ No flight data retrieved for any NYC airport.")
        return pd.DataFrame()

    combined = pd.concat(all_flights, ignore_index=True)
    print(f"\n📊 Combined: {len(combined)} flights across {len(all_flights)} airports")
    return combined


def compute_flight_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily/weekly flight frequency by airline for demand analysis.

    This is the key metric for M01: more flights = more capacity = demand signal.
    """
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
    print(f"Saved → {path}")
    return path


def load_aviation_data(filename: str) -> pd.DataFrame:
    path = DATA_RAW / "aviation_edge" / filename
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return pd.read_csv(path)


# ═══════════════════════════════════════════════════════════════
# SYNTHETIC DATA (for dev without API key)
# ═══════════════════════════════════════════════════════════════

def generate_synthetic_flight_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic NYC → DXB flight schedule data.

    Based on real patterns:
      - Emirates: 2-3 daily flights JFK→DXB (A380 + 777)
      - Emirates: 1 daily EWR→DXB
      - JetBlue codeshare / Delta seasonal
      - No LGA international

    FIX Bug A1: Uses STABLE flight numbers per airline-airport-slot
    FIX Bug A2: Calculates realistic arrival_scheduled (~14h later)
    """
    np.random.seed(seed)

    # ── FIX A1: Pre-assign stable flight numbers per route-slot ──
    airlines = {
        "EK": {
            "name": "Emirates",
            "airports": {
                "JFK": [
                    {"flight_num": "EK204", "dep_time": "02:15", "aircraft": "A380"},
                    {"flight_num": "EK202", "dep_time": "10:45", "aircraft": "B77W"},
                    {"flight_num": "EK206", "dep_time": "23:55", "aircraft": "A380"},
                ],
                "EWR": [
                    {"flight_num": "EK222", "dep_time": "22:30", "aircraft": "B77W"},
                ],
            },
        },
        "DL": {
            "name": "Delta Air Lines",
            "airports": {
                "JFK": [
                    {"flight_num": "DL420", "dep_time": "14:00", "aircraft": "A359"},
                ],
            },
        },
        "B6": {
            "name": "JetBlue Airways",
            "airports": {
                "JFK": [
                    {"flight_num": "B6725", "dep_time": "00:30", "aircraft": "A321"},
                ],
            },
        },
    }

    records = []
    dates = pd.date_range("2025-01-01", periods=90, freq="D")

    for date in dates:
        for airline_code, info in airlines.items():
            for airport, slots in info["airports"].items():
                for slot in slots:
                    # Seasonal variation: fewer flights in summer
                    if date.month in [6, 7, 8] and np.random.random() < 0.3:
                        continue

                    dep_time = slot["dep_time"]

                    # FIX A2: Calculate realistic arrival time (~14h flight)
                    dep_hour, dep_min = map(int, dep_time.split(":"))
                    arr_hour = (dep_hour + 14) % 24
                    next_day = (dep_hour + 14) >= 24
                    arr_date = date + pd.Timedelta(days=1) if next_day else date
                    arr_time = f"{arr_hour:02d}:{dep_min:02d}"

                    records.append({
                        "flight_iata": slot["flight_num"],
                        "airline_iata": airline_code,
                        "airline_name": info["name"],
                        "departure_airport": airport,
                        "departure_scheduled": f"{date.strftime('%Y-%m-%d')}T{dep_time}",
                        "departure_terminal": np.random.choice(["1", "4", "7"]),
                        "arrival_airport": "DXB",
                        "arrival_scheduled": f"{arr_date.strftime('%Y-%m-%d')}T{arr_time}",
                        "arrival_terminal": np.random.choice(["1", "3"]),
                        "status": "scheduled",
                        "aircraft_iata": slot["aircraft"],
                        "date": date,
                    })

    df = pd.DataFrame(records)
    print(f"Generated synthetic flight data: {len(df)} flights over {len(dates)} days")
    return df


def generate_synthetic_monthly_capacity(seed: int = 42) -> pd.DataFrame:
    """
    Generate monthly flight capacity time series (2019–2025).
    Key input for demand forecasting — seats available = supply side.
    """
    np.random.seed(seed)

    dates = pd.date_range("2019-01-01", "2025-12-01", freq="MS")

    seasonal = {
        1: 1.10, 2: 1.05, 3: 1.00, 4: 0.85, 5: 0.80, 6: 0.70,
        7: 0.65, 8: 0.70, 9: 0.85, 10: 1.00, 11: 1.10, 12: 1.20,
    }

    covid_factor = {
        2019: 1.0, 2020: 0.30, 2021: 0.55, 2022: 0.85,
        2023: 1.05, 2024: 1.15, 2025: 1.20,
    }

    records = []
    for date in dates:
        base_daily_flights = 5
        factor = seasonal[date.month] * covid_factor.get(date.year, 1.0)
        daily_flights = base_daily_flights * factor * np.random.uniform(0.85, 1.15)

        avg_seats_per_flight = 380
        monthly_days = pd.Period(date, "M").days_in_month

        monthly_flights = int(daily_flights * monthly_days)
        monthly_seats = int(daily_flights * avg_seats_per_flight * monthly_days)

        load_factor = np.random.uniform(0.75, 0.90)
        monthly_passengers = int(monthly_seats * load_factor)

        records.append({
            "DATE": date,
            "YEAR": date.year,
            "MONTH": date.month,
            "MONTHLY_FLIGHTS": monthly_flights,
            "MONTHLY_SEATS": monthly_seats,
            "LOAD_FACTOR": round(load_factor, 3),
            "EST_PASSENGERS": monthly_passengers,
            "AVG_DAILY_FLIGHTS": round(daily_flights, 1),
        })

    df = pd.DataFrame(records)
    print(f"Generated monthly capacity: {len(df)} months "
          f"({dates[0].strftime('%Y-%m')} → {dates[-1].strftime('%Y-%m')})")
    return df


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
            print("\n📊 Airline Frequency:")
            print(freq.to_string(index=False))
    else:
        print("No API key — generating synthetic data...\n")

        flights = generate_synthetic_flight_data()
        save_aviation_data(flights, "synthetic_flights_nyc_dxb.csv")

        capacity = generate_synthetic_monthly_capacity()
        save_aviation_data(capacity, "synthetic_monthly_capacity.csv")

        freq = compute_flight_frequency(flights)
        print("\n📊 Airline Frequency (synthetic):")
        print(freq.to_string(index=False))