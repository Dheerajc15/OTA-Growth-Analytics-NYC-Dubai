
from __future__ import annotations

from pathlib import Path
from typing import Optional

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
    AVIATION_EDGE_ENDPOINTS = {"routes": "/routes", "flights": "/flights"}
    ORIGIN_AIRPORTS = ["JFK", "EWR", "LGA"]
    DESTINATION_AIRPORT = "DXB"
    DATA_RAW = Path("data/raw")


def _check_key() -> bool:
    if not AVIATION_EDGE_API_KEY:
        print("⚠️ AVIATION_EDGE_API_KEY not set")
        return False
    return True


def _safe_get(endpoint: str, params: dict) -> list | dict:
    url = f"{AVIATION_EDGE_BASE_URL}{endpoint}"
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"API error {endpoint}: {e}")
        return []


def fetch_routes_data(limit: int = 5000) -> pd.DataFrame:
    if not _check_key():
        return pd.DataFrame()

    endpoint = AVIATION_EDGE_ENDPOINTS.get("routes", "/routes")
    payload = _safe_get(endpoint, {"key": AVIATION_EDGE_API_KEY, "limit": limit})

    if not isinstance(payload, list):
        return pd.DataFrame()

    df = pd.DataFrame(payload)
    if df.empty:
        return df

    origin_candidates = ["departureIata", "departure", "from"]
    dest_candidates = ["arrivalIata", "arrival", "to"]
    o_col = next((c for c in origin_candidates if c in df.columns), None)
    d_col = next((c for c in dest_candidates if c in df.columns), None)

    if o_col and d_col:
        df = df[df[o_col].isin(ORIGIN_AIRPORTS) & (df[d_col] == DESTINATION_AIRPORT)].copy()

    return df.reset_index(drop=True)


def fetch_aviation_data() -> pd.DataFrame:
    """
    Canonical fetch function used by notebooks.
    Returns monthly proxy if route list available.
    """
    routes = fetch_routes_data()
    if routes.empty:
        return pd.DataFrame()

    start = pd.Timestamp("2019-01-01")
    end = pd.Timestamp.today().normalize()
    months = pd.date_range(start=start, end=end, freq="MS")

    n_routes = max(len(routes), 1)
    out = pd.DataFrame({"DATE": months})
    out["MONTHLY_FLIGHTS"] = (n_routes * 25).astype(int)
    out["EST_PASSENGERS"] = (out["MONTHLY_FLIGHTS"] * 280 * 0.78).astype(int)
    out["LOAD_FACTOR"] = 0.78
    return out


# Backward-compatible aliases
fetch_route_data = fetch_routes_data
fetch_routes = fetch_routes_data


def save_aviation_data(df: pd.DataFrame, name: str = "monthly_capacity") -> Path:
    out_dir = DATA_RAW / "aviation_edge"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved → {path}")
    return path


def load_aviation_data(name: str = "monthly_capacity") -> pd.DataFrame:
    path = DATA_RAW / "aviation_edge" / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return pd.read_csv(path, parse_dates=["DATE"])

