"""
Centralized configuration for OTA Growth Analytics.
All paths, API keys, and project constants.

Data Sources:
  1. Google Places API — Dubai        
  2. Google Places API — NYC       
  3. Google Trends (pytrends)      
  4. Aviation Edge API               
  5. YouTube Data API v3            
  6. Simulated Fare Data (numpy)      
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Project root ───────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load .env explicitly from project root (fixes notebook CWD issues)
load_dotenv(ROOT_DIR / ".env")

# ═══════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════════
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
OUTPUTS = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS / "figures"
REPORTS_DIR = OUTPUTS / "reports"

# Ensure key dirs exist (safe no-op if already present)
for _p in [DATA_RAW, DATA_PROCESSED, OUTPUTS, FIGURES_DIR, REPORTS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# API KEYS
# ═══════════════════════════════════════════════════════════════
GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY", "").strip() or None
AVIATION_EDGE_API_KEY = os.getenv("AVIATION_EDGE_API_KEY", "").strip() or None

# ═══════════════════════════════════════════════════════════════
# ROUTE CONSTANTS
# ═══════════════════════════════════════════════════════════════
ORIGIN_AIRPORTS = ["JFK", "EWR", "LGA"]  # NYC area airports
DESTINATION_AIRPORT = "DXB"              # Dubai International
ORIGIN_IATA_CITY = "NYC"
DESTINATION_IATA_CITY = "DXB"
ROUTE_NAME = "NYC-Dubai"

# ═══════════════════════════════════════════════════════════════
# DATA SOURCE 1 & 2: GOOGLE PLACES API
# ═══════════════════════════════════════════════════════════════
GOOGLE_PLACES_BASE_URL = "https://maps.googleapis.com/maps/api/place"

PLACES_DUBAI_QUERIES = [
    "luxury hotels Dubai",
    "budget hotels Dubai",
    "hotels Dubai Marina",
    "hotels Downtown Dubai",
    "hotels Palm Jumeirah",
    "hotels Deira Dubai",
    "hotels JBR Dubai",
    "business hotels Dubai",
    "resorts Dubai",
    "hotels Dubai Creek",
]

PLACES_NYC_QUERIES = [
    "luxury hotels Manhattan",
    "budget hotels New York City",
    "hotels Times Square NYC",
    "hotels Brooklyn NYC",
    "hotels Midtown Manhattan",
    "hotels Lower Manhattan",
    "business hotels NYC",
    "boutique hotels NYC",
    "hotels near JFK airport",
    "hotels near Central Park",
]

DUBAI_LAT = 25.2048
DUBAI_LNG = 55.2708
NYC_LAT = 40.7128
NYC_LNG = -74.0060

# ═══════════════════════════════════════════════════════════════
# DATA SOURCE 3: GOOGLE TRENDS
# ═══════════════════════════════════════════════════════════════
TRENDS_KEYWORDS = [
    "NYC to Dubai flights",
    "Dubai hotels",
    "Dubai visa",
    "Dubai tourism",
    "cheap flights to Dubai",
]
TRENDS_TIMEFRAME = "2019-01-01 2025-12-31"
TRENDS_GEO = "US-NY"

# ═══════════════════════════════════════════════════════════════
# DATA SOURCE 4: AVIATION EDGE API
# ═══════════════════════════════════════════════════════════════
AVIATION_EDGE_BASE_URL = "https://aviation-edge.com/v2/public"
AVIATION_EDGE_ENDPOINTS = {
    "routes": "/routes",
    "flights": "/flights",
    "timetable": "/timetable",
    "airlines": "/airlineDatabase",
    "airports": "/airportDatabase",
}

# ═══════════════════════════════════════════════════════════════
# DATA SOURCE 5: YOUTUBE DATA API v3
# ═══════════════════════════════════════════════════════════════
YOUTUBE_SEARCH_QUERIES = [
    "NYC to Dubai travel vlog",
    "Dubai travel guide from New York",
    "flying to Dubai from JFK",
    "Dubai hotel review tourist",
    "Dubai things to do American tourist",
    "Dubai trip budget from USA",
]
YOUTUBE_MAX_RESULTS_PER_QUERY = 50

# ═══════════════════════════════════════════════════════════════
# DATA SOURCE 6: SIMULATED FARE DATA (M03)
# ═══════════════════════════════════════════════════════════════
FARE_RANGES = {
    "economy":  {"min": 400,  "max": 900,   "mean": 620,  "std": 130},
    "business": {"min": 2500, "max": 6000,  "mean": 3800, "std": 900},
    "first":    {"min": 8000, "max": 20000, "mean": 12000, "std": 3000},
}

# ═══════════════════════════════════════════════════════════════
# A/B TEST PARAMETERS (M03)
# ═══════════════════════════════════════════════════════════════
AB_TEST_SAMPLE_SIZE = 10000
AB_TEST_SEED = 42

# ═══════════════════════════════════════════════════════════════
# TRAVELER ARCHETYPES (M04)
# ═══════════════════════════════════════════════════════════════
TRAVELER_ARCHETYPES = {
    "business": {"stay_range": (2, 5), "fare_class": "business"},
    "leisure": {"stay_range": (5, 14), "fare_class": "economy"},
    "transit": {"stay_range": (1, 2), "fare_class": "economy"},
}

# ═══════════════════════════════════════════════════════════════
# VISA POLICY TIMELINE — UAE for US passport holders (M06)
# ═══════════════════════════════════════════════════════════════
VISA_POLICY_EVENTS = [
    {"date": "2014-01-01", "event": "Visa-on-arrival for US citizens (30-day free)", "friction_score": 2, "direction": "reduced"},
    {"date": "2017-06-01", "event": "6-month multiple entry visa option", "friction_score": 1, "direction": "reduced"},
    {"date": "2020-03-15", "event": "COVID-19 travel ban", "friction_score": 10, "direction": "increased"},
    {"date": "2020-07-07", "event": "UAE reopened with PCR test requirement", "friction_score": 7, "direction": "reduced"},
    {"date": "2022-02-26", "event": "All COVID entry requirements removed", "friction_score": 1, "direction": "reduced"},
    {"date": "2023-01-01", "event": "UAE visa-free expanded, streamlined e-gate", "friction_score": 1, "direction": "reduced"},
]