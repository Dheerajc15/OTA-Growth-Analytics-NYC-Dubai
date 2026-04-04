"""
Traveler Profile Generation (Preprocessing)
=============================================
Generates synthetic traveler profiles from hotel preference data.
This is a preprocessing step because it transforms hotel data into
a traveler-level dataset for downstream clustering.
"""

import numpy as np
import pandas as pd

try:
    from config.settings import TRAVELER_ARCHETYPES, FARE_RANGES, AB_TEST_SEED
except ImportError:
    TRAVELER_ARCHETYPES = {
        "business": {"stay_range": (2, 5), "fare_class": "business"},
        "leisure": {"stay_range": (5, 14), "fare_class": "economy"},
        "transit": {"stay_range": (1, 2), "fare_class": "economy"},
    }
    FARE_RANGES = {
        "economy": {"min": 400, "max": 900, "mean": 620, "std": 130},
        "business": {"min": 2500, "max": 6000, "mean": 3800, "std": 900},
        "first": {"min": 8000, "max": 20000, "mean": 12000, "std": 3000},
    }
    AB_TEST_SEED = 42


def generate_traveler_profiles(
    hotel_df: pd.DataFrame,
    n_travelers: int = 2000,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate synthetic traveler profiles linked to hotel preference data.

    Each traveler has:
      - Demographics: age, group_size, nationality_region
      - Trip behavior: stay_nights, lead_time_days, trip_purpose
      - Spending: flight_spend, hotel_spend_per_night, total_spend
      - Preferences: preferred_price_tier, preferred_market, device
      - Booking: booking_channel, loyalty_member, repeat_visitor
    """
    seed = seed or AB_TEST_SEED
    rng = np.random.RandomState(seed)

    purposes = rng.choice(
        ["leisure", "business", "transit", "honeymoon", "family_vacation"],
        size=n_travelers,
        p=[0.40, 0.30, 0.08, 0.12, 0.10],
    )

    # ── Age ──
    ages = np.zeros(n_travelers, dtype=int)
    for purpose, (low, high, mean, std) in {
        "leisure": (21, 65, 35, 10),
        "business": (25, 60, 40, 8),
        "transit": (20, 55, 32, 9),
        "honeymoon": (24, 40, 29, 4),
        "family_vacation": (28, 55, 38, 7),
    }.items():
        mask = purposes == purpose
        count = mask.sum()
        if count > 0:
            ages[mask] = np.clip(rng.normal(mean, std, size=count).astype(int), low, high)

    # ── Group size ──
    group_sizes = np.ones(n_travelers, dtype=int)
    group_sizes[purposes == "honeymoon"] = 2
    group_sizes[purposes == "family_vacation"] = rng.choice(
        [3, 4, 5], size=(purposes == "family_vacation").sum(), p=[0.4, 0.4, 0.2],
    )
    group_sizes[purposes == "business"] = rng.choice(
        [1, 2], size=(purposes == "business").sum(), p=[0.8, 0.2],
    )
    group_sizes[purposes == "leisure"] = rng.choice(
        [1, 2, 3, 4], size=(purposes == "leisure").sum(), p=[0.3, 0.4, 0.2, 0.1],
    )

    # ── Stay nights ──
    stay_nights = np.zeros(n_travelers, dtype=int)
    for purpose, archetype_key in {"business": "business", "transit": "transit"}.items():
        mask = purposes == purpose
        lo, hi = TRAVELER_ARCHETYPES[archetype_key]["stay_range"]
        stay_nights[mask] = rng.randint(lo, hi + 1, size=mask.sum())
    stay_nights[purposes == "leisure"] = rng.randint(5, 15, size=(purposes == "leisure").sum())
    stay_nights[purposes == "honeymoon"] = rng.randint(5, 11, size=(purposes == "honeymoon").sum())
    stay_nights[purposes == "family_vacation"] = rng.randint(6, 13, size=(purposes == "family_vacation").sum())

    # ── Lead time ──
    lead_times = np.zeros(n_travelers, dtype=int)
    lead_cfg = {
        "business": (14, 1, 90),
        "leisure": (45, 7, 180),
        "honeymoon": (60, 30, 200),
        "family_vacation": (50, 14, 180),
        "transit": (10, 1, 60),
    }
    for purpose, (scale, lo, hi) in lead_cfg.items():
        mask = purposes == purpose
        count = mask.sum()
        if count > 0:
            lead_times[mask] = np.clip(rng.exponential(scale, size=count), lo, hi).astype(int)

    # ── Flight spend ──
    fare_classes = np.full(n_travelers, "economy", dtype=object)
    fare_classes[purposes == "business"] = rng.choice(
        ["business", "economy", "first"], size=(purposes == "business").sum(), p=[0.55, 0.35, 0.10],
    )
    fare_classes[purposes == "honeymoon"] = rng.choice(
        ["business", "economy"], size=(purposes == "honeymoon").sum(), p=[0.40, 0.60],
    )

    flight_spend = np.zeros(n_travelers)
    for fc, params in FARE_RANGES.items():
        mask = fare_classes == fc
        count = mask.sum()
        if count > 0:
            flight_spend[mask] = np.clip(rng.normal(params["mean"], params["std"], size=count), params["min"], params["max"])

    # ── Hotel spend per night ──
    price_tier_prefs = np.full(n_travelers, "Mid-Range", dtype=object)
    price_tier_map = {
        "business": (["Upscale", "Luxury", "Mid-Range"], [0.45, 0.35, 0.20]),
        "leisure": (["Mid-Range", "Budget", "Upscale"], [0.45, 0.30, 0.25]),
        "transit": (["Budget", "Mid-Range"], [0.70, 0.30]),
        "honeymoon": (["Luxury", "Upscale"], [0.60, 0.40]),
        "family_vacation": (["Mid-Range", "Upscale", "Budget"], [0.50, 0.30, 0.20]),
    }
    hotel_nightly_rates = {
        "Budget": (80, 150, 110, 25),
        "Mid-Range": (150, 350, 230, 55),
        "Upscale": (300, 700, 450, 100),
        "Luxury": (500, 2000, 900, 250),
    }

    hotel_spend_per_night = np.zeros(n_travelers)
    for purpose, (tiers, probs) in price_tier_map.items():
        mask = purposes == purpose
        count = mask.sum()
        if count > 0:
            chosen_tiers = rng.choice(tiers, size=count, p=probs)
            price_tier_prefs[mask] = chosen_tiers
            for tier in np.unique(chosen_tiers):
                tier_mask = mask & (price_tier_prefs == tier)
                tc = tier_mask.sum()
                if tc > 0:
                    lo, hi, mean, std = hotel_nightly_rates[tier]
                    hotel_spend_per_night[tier_mask] = np.clip(rng.normal(mean, std, size=tc), lo, hi)

    total_hotel_spend = hotel_spend_per_night * stay_nights

    preferred_market = np.where(
        purposes == "transit", "NYC",
        rng.choice(["Dubai", "NYC"], size=n_travelers, p=[0.65, 0.35]),
    )
    devices = rng.choice(["mobile", "desktop", "tablet"], size=n_travelers, p=[0.52, 0.38, 0.10])
    channels = rng.choice(
        ["direct_web", "ota_app", "ota_web", "travel_agent", "metasearch"],
        size=n_travelers, p=[0.15, 0.30, 0.25, 0.15, 0.15],
    )
    loyalty_member = rng.choice([True, False], size=n_travelers, p=[0.25, 0.75])
    loyalty_member[purposes == "business"] = rng.choice(
        [True, False], size=(purposes == "business").sum(), p=[0.55, 0.45],
    )
    repeat_visitor = rng.choice([True, False], size=n_travelers, p=[0.15, 0.85])
    nationalities = rng.choice(["US_Northeast", "US_Other", "International"], size=n_travelers, p=[0.55, 0.30, 0.15])
    booking_dow = rng.choice(["weekday", "weekend"], size=n_travelers, p=[0.62, 0.38])
    booking_dow[purposes == "business"] = rng.choice(
        ["weekday", "weekend"], size=(purposes == "business").sum(), p=[0.82, 0.18],
    )

    df = pd.DataFrame({
        "TRAVELER_ID": [f"T{i:05d}" for i in range(n_travelers)],
        "AGE": ages,
        "GROUP_SIZE": group_sizes,
        "NATIONALITY_REGION": nationalities,
        "TRIP_PURPOSE": purposes,
        "STAY_NIGHTS": stay_nights,
        "LEAD_TIME_DAYS": lead_times,
        "FARE_CLASS": fare_classes,
        "FLIGHT_SPEND": flight_spend.round(2),
        "PREFERRED_PRICE_TIER": price_tier_prefs,
        "HOTEL_SPEND_PER_NIGHT": hotel_spend_per_night.round(2),
        "TOTAL_HOTEL_SPEND": total_hotel_spend.round(2),
        "TOTAL_TRIP_SPEND": (flight_spend + total_hotel_spend).round(2),
        "PREFERRED_MARKET": preferred_market,
        "DEVICE": devices,
        "BOOKING_CHANNEL": channels,
        "BOOKING_DOW": booking_dow,
        "LOYALTY_MEMBER": loyalty_member,
        "REPEAT_VISITOR": repeat_visitor,
    })

    print(f"Generated {len(df):,} traveler profiles")
    print(f"  Purposes: {dict(df['TRIP_PURPOSE'].value_counts())}")
    return df
