"""
Centralized configuration for OTA Growth Analytics.
All paths, API keys, and project constants live here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables from .env ───────────────────
load_dotenv()

# ── Project Paths ──────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
OUTPUTS = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS / "figures"
REPORTS_DIR = OUTPUTS / "reports"

# ── API Keys ───────────────────────────────────────────────
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "OTA-Growth-Analytics/1.0")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

# ── Route Constants ────────────────────────────────────────
ORIGIN_AIRPORTS = ["JFK", "EWR", "LGA"]       # NYC area airports
DESTINATION_AIRPORT = "DXB"                     # Dubai International
ROUTE_NAME = "NYC-Dubai"

# ── Google Trends Keywords ─────────────────────────────────
TRENDS_KEYWORDS = [
    "NYC to Dubai flights",
    "Dubai hotels",
    "Dubai visa",
    "Dubai tourism",
    "cheap flights to Dubai",
]

# ── BTS Data Settings ──────────────────────────────────────
# Download from: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoession_ID=0&Table_ID=272
BTS_YEARS = list(range(2018, 2026))
BTS_QUARTERS = [1, 2, 3, 4]

# ── Inside Airbnb URLs ─────────────────────────────────────
# Check http://insideairbnb.com/get-the-data/ for latest URLs
AIRBNB_NYC_URL = (
    "http://data.insideairbnb.com/united-states/ny/"
    "new-york-city/2024-01-05/data/listings.csv.gz"
)
AIRBNB_DUBAI_URL = (
    "http://data.insideairbnb.com/united-arab-emirates/"
    "dubai/dubai/2024-01-15/data/listings.csv.gz"
)

# ── A/B Test Parameters ────────────────────────────────────
AB_TEST_SAMPLE_SIZE = 10000
AB_TEST_SEED = 42

# ── Traveler Archetypes ────────────────────────────────────
TRAVELER_ARCHETYPES = {
    "business":  {"stay_range": (2, 5),  "fare_class": "business"},
    "leisure":   {"stay_range": (5, 14), "fare_class": "economy"},
    "transit":   {"stay_range": (1, 2),  "fare_class": "economy"},
}

# ── Fare Ranges (JFK → DXB, from BTS DB1B historical data) ─
FARE_RANGES = {
    "economy":  {"min": 400,  "max": 900,   "mean": 620,  "std": 130},
    "business": {"min": 2500, "max": 6000,  "mean": 3800, "std": 900},
    "first":    {"min": 8000, "max": 20000, "mean": 12000, "std": 3000},
}