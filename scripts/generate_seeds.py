
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from config.settings import (
        DUBAI_LAT, DUBAI_LNG, NYC_LAT, NYC_LNG,
        FARE_RANGES, AB_TEST_SAMPLE_SIZE, AB_TEST_SEED,
        TRAVELER_ARCHETYPES,
    )
except ImportError:
    DUBAI_LAT, DUBAI_LNG = 25.2048, 55.2708
    NYC_LAT, NYC_LNG = 40.7128, -74.0060
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

SEEDS_DIR = ROOT / "data" / "seeds"


# ═══════════════════════════════════════════════════════════════
# GOOGLE TRENDS (weekly, 2019–2025)
# ═══════════════════════════════════════════════════════════════

def generate_trends(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2019-01-06", "2025-12-28", freq="W-SUN")
    n = len(dates)

    base_levels = {
        "NYC to Dubai flights": 35,
        "Dubai hotels": 50,
        "Dubai visa": 25,
        "Dubai tourism": 40,
        "cheap flights to Dubai": 30,
    }
    seasonal_map = {
        1: 1.30, 2: 1.10, 3: 0.95, 4: 0.85, 5: 0.75, 6: 0.70,
        7: 0.60, 8: 0.65, 9: 0.80, 10: 0.95, 11: 1.15, 12: 1.40,
    }

    data = {}
    for keyword, base in base_levels.items():
        values = np.zeros(n)
        for i, date in enumerate(dates):
            seasonal = seasonal_map[date.month]
            covid = 1.0
            if pd.Timestamp("2020-03-01") <= date <= pd.Timestamp("2020-06-30"):
                covid = 0.2
            elif pd.Timestamp("2020-07-01") <= date <= pd.Timestamp("2021-06-30"):
                covid = 0.5
            elif pd.Timestamp("2021-07-01") <= date <= pd.Timestamp("2022-03-31"):
                covid = 0.75
            growth = 1 + (date.year - 2019) * 0.04
            value = base * seasonal * covid * growth
            value += np.random.normal(0, base * 0.12)
            values[i] = max(0, min(100, value))
        data[keyword] = np.round(values).astype(int)

    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


# ═══════════════════════════════════════════════════════════════
# AVIATION EDGE — FLIGHT SCHEDULE (90-day)
# ═══════════════════════════════════════════════════════════════

def generate_flights(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

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
                    if date.month in [6, 7, 8] and np.random.random() < 0.3:
                        continue
                    dep_time = slot["dep_time"]
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

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# AVIATION EDGE — MONTHLY CAPACITY (2019–2025)
# ═══════════════════════════════════════════════════════════════

def generate_capacity(seed: int = 42) -> pd.DataFrame:
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
        base_daily = 5
        factor = seasonal[date.month] * covid_factor.get(date.year, 1.0)
        daily_flights = base_daily * factor * np.random.uniform(0.85, 1.15)
        avg_seats = 380
        days_in_month = pd.Period(date, "M").days_in_month
        monthly_flights = int(daily_flights * days_in_month)
        monthly_seats = int(daily_flights * avg_seats * days_in_month)
        load_factor = np.random.uniform(0.75, 0.90)
        est_passengers = int(monthly_seats * load_factor)

        records.append({
            "DATE": date,
            "YEAR": date.year,
            "MONTH": date.month,
            "MONTHLY_FLIGHTS": monthly_flights,
            "MONTHLY_SEATS": monthly_seats,
            "LOAD_FACTOR": round(load_factor, 3),
            "EST_PASSENGERS": est_passengers,
            "AVG_DAILY_FLIGHTS": round(daily_flights, 1),
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# GOOGLE PLACES — HOTELS (Dubai + NYC)
# ═══════════════════════════════════════════════════════════════

def generate_hotels(seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    # ── Dubai ──
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

    dubai_records = []
    for i in range(n_dubai):
        neighborhood = rng.choice(dubai_neighborhoods)
        name = f"{rng.choice(dubai_prefixes)} {neighborhood}"
        price_level = rng.choice([1, 2, 3, 4], p=[0.05, 0.20, 0.40, 0.35])
        rating = round(float(np.clip(rng.normal(4.3, 0.4), 2.5, 5.0)), 1)
        total_ratings = int(rng.lognormal(7.0, 1.2))
        num_reviews = min(5, max(0, int(rng.normal(4, 1.5))))

        dubai_records.append({
            "PLACE_ID": f"SYNTH_DUBAI_{i:04d}",
            "NAME": name,
            "MARKET": "Dubai",
            "RATING": rating,
            "TOTAL_RATINGS": total_ratings,
            "PRICE_LEVEL": price_level,
            "ADDRESS": f"{neighborhood}, Dubai, UAE",
            "LAT": DUBAI_LAT + rng.uniform(-0.08, 0.08),
            "LNG": DUBAI_LNG + rng.uniform(-0.08, 0.08),
            "BUSINESS_STATUS": rng.choice(["OPERATIONAL", "CLOSED_TEMPORARILY"], p=[0.98, 0.02]),
            "TYPES": "lodging, point_of_interest, establishment",
            "NEIGHBORHOOD": neighborhood,
            "NUM_PHOTOS": rng.randint(3, 30),
            "NUM_REVIEWS_FETCHED": num_reviews,
            "WEBSITE": f"https://www.{name.lower().replace(' ', '').replace('&', '')}.com",
            "REVIEW_TEXTS": " ||| ".join(rng.choice(dubai_sentiments, size=num_reviews)) if num_reviews else "",
        })

    # ── NYC ──
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

    nyc_records = []
    for i in range(n_nyc):
        neighborhood = rng.choice(nyc_neighborhoods)
        name = f"{rng.choice(nyc_prefixes)} {neighborhood}"
        price_level = rng.choice([1, 2, 3, 4], p=[0.15, 0.40, 0.30, 0.15])
        rating = round(float(np.clip(rng.normal(4.0, 0.5), 2.0, 5.0)), 1)
        total_ratings = int(rng.lognormal(7.5, 1.3))
        num_reviews = min(5, max(0, int(rng.normal(4, 1))))
        clean_name = name.lower().replace(" ", "").replace("'", "")
        website = f"https://www.{clean_name}.com" if rng.random() < 0.75 else ""

        nyc_records.append({
            "PLACE_ID": f"SYNTH_NYC_{i:04d}",
            "NAME": name,
            "MARKET": "NYC",
            "RATING": rating,
            "TOTAL_RATINGS": total_ratings,
            "PRICE_LEVEL": price_level,
            "ADDRESS": f"{neighborhood}, New York, NY, USA",
            "LAT": NYC_LAT + rng.uniform(-0.06, 0.06),
            "LNG": NYC_LNG + rng.uniform(-0.06, 0.06),
            "BUSINESS_STATUS": "OPERATIONAL",
            "TYPES": "lodging, point_of_interest, establishment",
            "NEIGHBORHOOD": neighborhood,
            "NUM_PHOTOS": rng.randint(2, 25),
            "NUM_REVIEWS_FETCHED": num_reviews,
            "WEBSITE": website,
            "REVIEW_TEXTS": " ||| ".join(rng.choice(nyc_sentiments, size=num_reviews)) if num_reviews else "",
        })

    combined = pd.DataFrame(dubai_records + nyc_records)

    # Synthetic average review rating
    avg_ratings = []
    for _, row in combined.iterrows():
        if row["NUM_REVIEWS_FETCHED"] > 0:
            avg_ratings.append(round(float(np.clip(row["RATING"] + rng.uniform(-0.3, 0.3), 1, 5)), 2))
        else:
            avg_ratings.append(None)
    combined["AVG_REVIEW_RATING"] = avg_ratings
    return combined


# ═══════════════════════════════════════════════════════════════
# YOUTUBE — TRAVEL VIDEOS (200 videos)
# ═══════════════════════════════════════════════════════════════

def generate_youtube(seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    themes = {
        "travel_vlog": {
            "weight": 0.25,
            "titles": [
                "NYC to Dubai FIRST CLASS on Emirates A380!",
                "I flew from JFK to Dubai for $400 (here's how)",
                "48 Hours in Dubai - NYC Girl's First Time!",
                "Dubai Travel Vlog 2024 | From New York to the Desert",
                "Flying Business Class NYC→Dubai | Is it worth $3,800?",
                "My INSANE Dubai Trip from New York City",
                "JFK to DXB: 14 Hours on Emirates (Full Review)",
                "NYC→Dubai: The Ultimate Travel Day Vlog",
            ],
            "avg_views": 150000, "view_std": 200000, "engagement_mult": 1.2,
        },
        "travel_guide": {
            "weight": 0.20,
            "titles": [
                "Dubai Travel Guide 2024: Everything You Need to Know",
                "First Time in Dubai? Watch This BEFORE You Go",
                "Dubai on a Budget: Tips from a New Yorker",
                "Top 20 Things to Do in Dubai (from an American)",
                "Dubai Travel Guide for US Citizens | Visa, Money, Safety",
                "What I Wish I Knew Before Going to Dubai from NYC",
            ],
            "avg_views": 250000, "view_std": 300000, "engagement_mult": 1.0,
        },
        "hotel_review": {
            "weight": 0.20,
            "titles": [
                "Atlantis the Palm Dubai - Honest Review (Worth $500/night?)",
                "I Stayed at the CHEAPEST Hotel in Dubai Marina",
                "Burj Al Arab vs Atlantis: Which Dubai Hotel is Better?",
                "Dubai Hotel Room Tour: Budget vs Luxury Comparison",
                "$50 vs $5,000 Hotel in Dubai - Is Luxury Worth It?",
                "Best Hotels in Dubai for American Tourists (2024)",
            ],
            "avg_views": 100000, "view_std": 150000, "engagement_mult": 1.1,
        },
        "things_to_do": {
            "weight": 0.15,
            "titles": [
                "Dubai Desert Safari - Is It a Tourist Trap?",
                "Top 10 FREE Things to Do in Dubai",
                "Dubai Mall Tour: World's Largest Shopping Mall!",
                "Trying Street Food in Old Dubai (Deira Market)",
                "Dubai Nightlife Guide for American Tourists",
                "Most UNDERRATED Things to Do in Dubai",
            ],
            "avg_views": 180000, "view_std": 250000, "engagement_mult": 1.3,
        },
        "budget_planning": {
            "weight": 0.10,
            "titles": [
                "How Much Does a Dubai Trip ACTUALLY Cost from NYC?",
                "Dubai Trip Budget Breakdown: 7 Days, $2,500 Total",
                "Cheap Flights from NYC to Dubai: My Strategy",
                "Is Dubai EXPENSIVE? Real Costs for Americans",
                "How to Visit Dubai on a NYC Budget",
            ],
            "avg_views": 80000, "view_std": 120000, "engagement_mult": 0.9,
        },
        "flight_review": {
            "weight": 0.10,
            "titles": [
                "Emirates vs Etihad: Which is Better from JFK?",
                "JFK to Dubai: Economy Class Honest Review",
                "Flying to Dubai from New York: What to Expect",
                "Emirates A380 Business Class Review (JFK→DXB)",
                "Best Airlines NYC to Dubai Ranked",
            ],
            "avg_views": 200000, "view_std": 280000, "engagement_mult": 1.15,
        },
    }

    channels = [
        {"name": "WanderLuxe NYC", "subs": 850000, "type": "luxury"},
        {"name": "Budget Backpacker", "subs": 420000, "type": "budget"},
        {"name": "The Points Guy Travel", "subs": 1200000, "type": "points"},
        {"name": "NYC Travel Diaries", "subs": 350000, "type": "local"},
        {"name": "Kara and Nate", "subs": 3500000, "type": "couple"},
        {"name": "Mark Wiens (Food)", "subs": 9500000, "type": "food"},
        {"name": "Lost LeBlanc", "subs": 2800000, "type": "cinematic"},
        {"name": "Hey Nadine", "subs": 750000, "type": "solo_female"},
        {"name": "Wolters World", "subs": 1600000, "type": "tips"},
        {"name": "Flying The Nest", "subs": 900000, "type": "family"},
        {"name": "Nomadic Matt", "subs": 500000, "type": "budget"},
        {"name": "Sorelle Amore", "subs": 1100000, "type": "lifestyle"},
        {"name": "Travel Tips by Dheeraj", "subs": 25000, "type": "small"},
        {"name": "Dubai Insider Guide", "subs": 180000, "type": "local"},
        {"name": "NY to Anywhere", "subs": 95000, "type": "small"},
    ]

    query_map = {
        "travel_vlog": "NYC to Dubai travel vlog",
        "travel_guide": "Dubai travel guide from New York",
        "hotel_review": "Dubai hotel review tourist",
        "things_to_do": "Dubai things to do American tourist",
        "budget_planning": "Dubai trip budget from USA",
        "flight_review": "flying to Dubai from JFK",
    }

    theme_names = list(themes.keys())
    theme_weights = [themes[t]["weight"] for t in theme_names]
    date_end = pd.Timestamp("2025-12-31")
    date_range_days = (date_end - pd.Timestamp("2019-01-01")).days

    records = []
    for i in range(200):
        theme_key = rng.choice(theme_names, p=theme_weights)
        theme = themes[theme_key]
        title = rng.choice(theme["titles"])
        year = rng.choice(["2023", "2024", "2025", ""])
        if year and rng.random() < 0.3:
            title = title.replace("2024", year)

        channel = rng.choice(channels)
        base_views = max(100, int(rng.lognormal(np.log(theme["avg_views"]), 0.8)))
        channel_mult = np.clip(channel["subs"] / 1_000_000, 0.1, 3.0)
        views = int(base_views * channel_mult)

        like_rate = np.clip(rng.normal(0.035, 0.015), 0.005, 0.10)
        comment_rate = np.clip(rng.normal(0.005, 0.003), 0.0005, 0.03)
        likes = int(views * like_rate * theme["engagement_mult"])
        comments = int(views * comment_rate * theme["engagement_mult"])

        if theme_key in ["travel_vlog", "flight_review"]:
            duration_min = int(np.clip(rng.normal(18, 6), 8, 45))
        elif theme_key == "travel_guide":
            duration_min = int(np.clip(rng.normal(22, 8), 10, 60))
        else:
            duration_min = int(np.clip(rng.normal(12, 5), 4, 35))

        days_ago = int(rng.exponential(365))
        pub_date = date_end - pd.Timedelta(days=min(days_ago, date_range_days))

        desc_templates = [
            f"In this video I share my experience {title.lower()}. Subscribe for more travel content!",
            f"Everything you need to know about traveling from NYC to Dubai. Links in description!",
            f"Join me on my journey from New York to Dubai! Don't forget to like and subscribe!",
        ]

        records.append({
            "VIDEO_ID": f"SYNTH_YT_{i:04d}",
            "TITLE": title,
            "DESCRIPTION": rng.choice(desc_templates),
            "CHANNEL_TITLE": channel["name"],
            "CHANNEL_ID": f"UC{rng.choice(list('ABCDEFGHIJKLMNOP'))}{i:06d}",
            "CHANNEL_SUBSCRIBERS": channel["subs"],
            "CHANNEL_TYPE": channel["type"],
            "PUBLISHED_AT": pub_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "SEARCH_QUERY": query_map.get(theme_key, "Dubai travel"),
            "CONTENT_THEME": theme_key,
            "VIEW_COUNT": views,
            "LIKE_COUNT": likes,
            "COMMENT_COUNT": comments,
            "DURATION_MINUTES": duration_min,
            "DURATION": f"PT{duration_min}M{rng.randint(0, 60)}S",
        })

    df = pd.DataFrame(records)
    df["PUBLISHED_AT"] = pd.to_datetime(df["PUBLISHED_AT"])
    return df


# ═══════════════════════════════════════════════════════════════
# A/B TEST DATA (10K visitors)
# ═══════════════════════════════════════════════════════════════

def generate_ab_test(
    n: int = None,
    seed: int = None,
    control_cvr: float = 0.032,
    treatment_lift: float = 0.35,
    bundle_discount: float = 0.12,
) -> pd.DataFrame:
    n = n or AB_TEST_SAMPLE_SIZE
    seed = seed or AB_TEST_SEED
    rng = np.random.RandomState(seed)

    n_control = n // 2
    n_treatment = n - n_control
    treatment_cvr = control_cvr * (1 + treatment_lift)

    fare_class_probs = {"economy": 0.70, "business": 0.25, "first": 0.05}
    fare_classes = rng.choice(list(fare_class_probs.keys()), size=n, p=list(fare_class_probs.values()))

    hotel_prices = np.zeros(n)
    for fc, params in FARE_RANGES.items():
        mask = fare_classes == fc
        count = mask.sum()
        if count > 0:
            prices = np.clip(rng.normal(params["mean"], params["std"], size=count), params["min"], params["max"])
            hotel_prices[mask] = prices

    traveler_types = rng.choice(["leisure", "business", "transit"], size=n, p=[0.55, 0.35, 0.10])
    devices = rng.choice(["mobile", "desktop", "tablet"], size=n, p=[0.55, 0.35, 0.10])
    lead_times = np.clip(rng.exponential(scale=25, size=n), 1, 180).astype(int)
    day_of_week = rng.choice(7, size=n, p=[0.13, 0.15, 0.14, 0.14, 0.16, 0.15, 0.13])

    groups = np.array(["control"] * n_control + ["treatment"] * n_treatment)
    bundle_prices = hotel_prices.copy()
    treatment_mask = groups == "treatment"
    bundle_prices[treatment_mask] *= (1 - bundle_discount)

    base_cvr = np.where(groups == "control", control_cvr, treatment_cvr)
    fare_mult = np.where(fare_classes == "business", 1.25, np.where(fare_classes == "first", 0.80, 1.0))
    device_mult = np.where(devices == "desktop", 1.15, np.where(devices == "mobile", 0.90, 1.0))
    lead_mult = np.where((lead_times >= 7) & (lead_times <= 30), 1.10, np.where(lead_times > 60, 0.85, 1.0))
    bundle_sens = np.where(
        (groups == "treatment") & (fare_classes == "economy"), 1.15,
        np.where((groups == "treatment") & (fare_classes == "first"), 0.95, 1.0),
    )
    effective_cvr = np.clip(base_cvr * fare_mult * device_mult * lead_mult * bundle_sens, 0.005, 0.20)
    converted = rng.binomial(1, effective_cvr)
    display_price = np.where(groups == "treatment", bundle_prices, hotel_prices)
    revenue = np.where(converted == 1, display_price, 0.0)

    return pd.DataFrame({
        "visitor_id": [f"V{i:06d}" for i in range(n)],
        "group": groups,
        "fare_class": fare_classes,
        "hotel_price": hotel_prices.round(2),
        "bundle_price": bundle_prices.round(2),
        "display_price": display_price.round(2),
        "traveler_type": traveler_types,
        "device": devices,
        "lead_time_days": lead_times,
        "day_of_week": day_of_week,
        "converted": converted,
        "revenue": revenue.round(2),
    })


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate seed data fixtures")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    ext = args.format

    generators = {
        "google_trends": generate_trends,
        "aviation_flights": generate_flights,
        "aviation_capacity": generate_capacity,
        "hotels": generate_hotels,
        "youtube": generate_youtube,
        "ab_test": generate_ab_test,
    }

    for name, fn in generators.items():
        print(f"Generating {name}...", end=" ")
        df = fn(seed=args.seed)
        path = SEEDS_DIR / f"{name}.{ext}"
        if ext == "parquet":
            df.to_parquet(path, index=True)
        else:
            df.to_csv(path)
        print(f"{len(df)} rows -> {path}")

    print(f"\nAll seeds written to {SEEDS_DIR}/")


if __name__ == "__main__":
    main()
