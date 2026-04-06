from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
DATA_SEEDS = ROOT_DIR / "data" / "seeds"
OUTPUTS = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS / "figures"
REPORTS_DIR = OUTPUTS / "reports"

for _p in [DATA_RAW, DATA_PROCESSED, DATA_SEEDS, OUTPUTS, FIGURES_DIR, REPORTS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY", "").strip() or None
AVIATION_EDGE_API_KEY = os.getenv("AVIATION_EDGE_API_KEY", "").strip() or None

ORIGIN_AIRPORTS = ["JFK", "EWR", "LGA"]
DESTINATION_AIRPORT = "DXB"
ROUTE_NAME = "NYC-Dubai"

GOOGLE_PLACES_BASE_URL = "https://maps.googleapis.com/maps/api/place"
PLACES_DUBAI_QUERIES = ["luxury hotels Dubai", "budget hotels Dubai"]
PLACES_NYC_QUERIES = ["luxury hotels Manhattan", "budget hotels New York City"]
DUBAI_LAT, DUBAI_LNG = 25.2048, 55.2708
NYC_LAT, NYC_LNG = 40.7128, -74.0060

TRENDS_KEYWORDS = ["NYC to Dubai flights", "Dubai hotels", "Dubai visa", "Dubai tourism"]
TRENDS_TIMEFRAME = "2019-01-01 2025-12-31"
TRENDS_GEO = "US-NY"

AVIATION_EDGE_BASE_URL = "https://aviation-edge.com/v2/public"
AVIATION_EDGE_ENDPOINTS = {"routes": "/routes"}

YOUTUBE_SEARCH_QUERIES = [
    "NYC to Dubai travel vlog",
    "Dubai travel guide from New York",
    "flying to Dubai from JFK",
]
YOUTUBE_MAX_RESULTS_PER_QUERY = 50

FARE_RANGES = {
    "economy": {"min": 400, "max": 900, "mean": 620, "std": 130},
    "business": {"min": 2500, "max": 6000, "mean": 3800, "std": 900},
    "first": {"min": 8000, "max": 20000, "mean": 12000, "std": 3000},
}
AB_TEST_SAMPLE_SIZE = 10000
AB_TEST_SEED = 42

TRAVELER_ARCHETYPES = {
    "business": {"stay_range": (2, 5), "fare_class": "business"},
    "leisure": {"stay_range": (5, 14), "fare_class": "economy"},
    "transit": {"stay_range": (1, 2), "fare_class": "economy"},
}

VISA_POLICY_EVENTS = [
    {"date": "2014-01-01", "event": "Visa-on-arrival", "friction_score": 2, "direction": "reduced"},
    {"date": "2020-03-15", "event": "COVID travel ban", "friction_score": 10, "direction": "increased"},
    {"date": "2022-02-26", "event": "COVID entry requirements removed", "friction_score": 1, "direction": "reduced"},
]