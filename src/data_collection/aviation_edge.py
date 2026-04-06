from __future__ import annotations

from pathlib import Path
import pandas as pd
import requests

try:
    from config.settings import (
        AVIATION_EDGE_API_KEY, AVIATION_EDGE_BASE_URL, AVIATION_EDGE_ENDPOINTS,
        ORIGIN_AIRPORTS, DESTINATION_AIRPORT, DATA_RAW
    )
except ImportError:
    AVIATION_EDGE_API_KEY = None
    AVIATION_EDGE_BASE_URL = "https://aviation-edge.com/v2/public"
    AVIATION_EDGE_ENDPOINTS = {"routes": "/routes"}
    ORIGIN_AIRPORTS = ["JFK", "EWR", "LGA"]
    DESTINATION_AIRPORT = "DXB"
    DATA_RAW = Path("data/raw")


def fetch_routes_data(limit: int = 5000) -> pd.DataFrame:
    if not AVIATION_EDGE_API_KEY:
        return pd.DataFrame()

    endpoint = AVIATION_EDGE_ENDPOINTS.get("routes", "/routes")
    try:
        r = requests.get(
            f"{AVIATION_EDGE_BASE_URL}{endpoint}",
            params={"key": AVIATION_EDGE_API_KEY, "limit": limit},
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return pd.DataFrame()

    if not isinstance(payload, list):
        return pd.DataFrame()

    df = pd.DataFrame(payload)
    if df.empty:
        return df

    o = next((c for c in ["departureIata", "departure", "from"] if c in df.columns), None)
    d = next((c for c in ["arrivalIata", "arrival", "to"] if c in df.columns), None)
    if o and d:
        df = df[df[o].isin(ORIGIN_AIRPORTS) & (df[d] == DESTINATION_AIRPORT)]
    return df.reset_index(drop=True)


def fetch_aviation_data() -> pd.DataFrame:
    # RAW collection module should not synthesize proxies
    return fetch_routes_data()


fetch_route_data = fetch_routes_data
fetch_routes = fetch_routes_data