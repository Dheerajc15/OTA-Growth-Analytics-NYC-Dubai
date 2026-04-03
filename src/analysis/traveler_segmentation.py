"""
Traveler Segmentation Engine (Module 04)
==========================================
Data Sources: Google Places API — Dubai (#1) + NYC (#2)

Business Question:
  "What distinct traveler archetypes exist on the NYC→Dubai route,
   and how should the OTA tailor products, pricing, and marketing
   to each segment?"

Approach:
  1. Generate synthetic traveler profiles from hotel interaction data
  2. Engineer behavioral features (spend, stay length, booking pattern)
  3. PCA dimensionality reduction → K-Means clustering
  4. Profile & label each cluster with business-meaningful archetypes
  5. Cross-tabulate segments with hotel market preferences
  6. Recommend segment-specific OTA strategies
"""

import os
import pandas as pd
import numpy as np
from typing import Optional
from scipy import stats

# ── Silence Windows wmic warning from joblib/loky ──
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ scikit-learn not installed — clustering features disabled")

try:
    from config.settings import (
        TRAVELER_ARCHETYPES, FARE_RANGES, AB_TEST_SEED,
    )
except ImportError:
    TRAVELER_ARCHETYPES = {
        "business":  {"stay_range": (2, 5),  "fare_class": "business"},
        "leisure":   {"stay_range": (5, 14), "fare_class": "economy"},
        "transit":   {"stay_range": (1, 2),  "fare_class": "economy"},
    }
    FARE_RANGES = {
        "economy":  {"min": 400,  "max": 900,   "mean": 620,  "std": 130},
        "business": {"min": 2500, "max": 6000,  "mean": 3800, "std": 900},
        "first":    {"min": 8000, "max": 20000, "mean": 12000, "std": 3000},
    }
    AB_TEST_SEED = 42


# ═══════════════════════════════════════════════════════════════
# SYNTHETIC TRAVELER PROFILE GENERATION
# ═══════════════════════════════════════════════════════════════

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

    Parameters
    ----------
    hotel_df : prepared hotel data (from prepare_funnel_data)
    n_travelers : number of traveler profiles to generate
    seed : random seed
    """
    seed = seed or AB_TEST_SEED
    rng = np.random.RandomState(seed)

    # ── Trip purpose distribution (NYC→Dubai specific) ──
    purposes = rng.choice(
        ["leisure", "business", "transit", "honeymoon", "family_vacation"],
        size=n_travelers,
        p=[0.40, 0.30, 0.08, 0.12, 0.10],
    )

    # ── Age distribution varies by purpose ──
    ages = np.zeros(n_travelers, dtype=int)
    for purpose, (low, high, mean, std) in {
        "leisure":          (21, 65, 35, 10),
        "business":         (25, 60, 40, 8),
        "transit":          (20, 55, 32, 9),
        "honeymoon":        (24, 40, 29, 4),
        "family_vacation":  (28, 55, 38, 7),
    }.items():
        mask = purposes == purpose
        count = mask.sum()
        if count > 0:
            ages[mask] = np.clip(
                rng.normal(mean, std, size=count).astype(int), low, high
            )

    # ── Group size ──
    group_sizes = np.ones(n_travelers, dtype=int)
    group_sizes[purposes == "honeymoon"] = 2
    group_sizes[purposes == "family_vacation"] = rng.choice(
        [3, 4, 5],
        size=(purposes == "family_vacation").sum(),
        p=[0.4, 0.4, 0.2],
    )
    group_sizes[purposes == "business"] = rng.choice(
        [1, 2],
        size=(purposes == "business").sum(),
        p=[0.8, 0.2],
    )
    group_sizes[purposes == "leisure"] = rng.choice(
        [1, 2, 3, 4],
        size=(purposes == "leisure").sum(),
        p=[0.3, 0.4, 0.2, 0.1],
    )

    # ── Stay duration (nights) ──
    stay_nights = np.zeros(n_travelers, dtype=int)
    for purpose, archetype_key in {
        "business": "business", "transit": "transit",
    }.items():
        mask = purposes == purpose
        lo, hi = TRAVELER_ARCHETYPES[archetype_key]["stay_range"]
        stay_nights[mask] = rng.randint(lo, hi + 1, size=mask.sum())

    stay_nights[purposes == "leisure"] = rng.randint(
        5, 15, size=(purposes == "leisure").sum()
    )
    stay_nights[purposes == "honeymoon"] = rng.randint(
        5, 11, size=(purposes == "honeymoon").sum()
    )
    stay_nights[purposes == "family_vacation"] = rng.randint(
        6, 13, size=(purposes == "family_vacation").sum()
    )

    # ── Booking lead time (days before departure) ──
    lead_times = np.zeros(n_travelers, dtype=int)
    lead_cfg = {
        "business":         (14, 1, 90),
        "leisure":          (45, 7, 180),
        "honeymoon":        (60, 30, 200),
        "family_vacation":  (50, 14, 180),
        "transit":          (10, 1, 60),
    }
    for purpose, (scale, lo, hi) in lead_cfg.items():
        mask = purposes == purpose
        count = mask.sum()
        if count > 0:
            lead_times[mask] = np.clip(
                rng.exponential(scale, size=count), lo, hi
            ).astype(int)

    # ── Flight spend (based on fare class × purpose) ──
    fare_classes = np.full(n_travelers, "economy", dtype=object)
    fare_classes[purposes == "business"] = rng.choice(
        ["business", "economy", "first"],
        size=(purposes == "business").sum(),
        p=[0.55, 0.35, 0.10],
    )
    fare_classes[purposes == "honeymoon"] = rng.choice(
        ["business", "economy"],
        size=(purposes == "honeymoon").sum(),
        p=[0.40, 0.60],
    )

    flight_spend = np.zeros(n_travelers)
    for fc, params in FARE_RANGES.items():
        mask = fare_classes == fc
        count = mask.sum()
        if count > 0:
            flight_spend[mask] = np.clip(
                rng.normal(params["mean"], params["std"], size=count),
                params["min"], params["max"],
            )

    # ── Hotel spend per night (tied to purpose + price tier preference) ──
    price_tier_prefs = np.full(n_travelers, "Mid-Range", dtype=object)
    price_tier_map = {
        "business":         (["Upscale", "Luxury", "Mid-Range"], [0.45, 0.35, 0.20]),
        "leisure":          (["Mid-Range", "Budget", "Upscale"], [0.45, 0.30, 0.25]),
        "transit":          (["Budget", "Mid-Range"], [0.70, 0.30]),
        "honeymoon":        (["Luxury", "Upscale"], [0.60, 0.40]),
        "family_vacation":  (["Mid-Range", "Upscale", "Budget"], [0.50, 0.30, 0.20]),
    }

    hotel_nightly_rates = {
        "Budget":    (80, 150, 110, 25),
        "Mid-Range": (150, 350, 230, 55),
        "Upscale":   (300, 700, 450, 100),
        "Luxury":    (500, 2000, 900, 250),
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
                    hotel_spend_per_night[tier_mask] = np.clip(
                        rng.normal(mean, std, size=tc), lo, hi
                    )

    total_hotel_spend = hotel_spend_per_night * stay_nights

    # ── Preferred market (Dubai = destination, NYC = origin stopover) ──
    preferred_market = np.where(
        purposes == "transit", "NYC",
        rng.choice(["Dubai", "NYC"], size=n_travelers, p=[0.65, 0.35]),
    )

    # ── Device ──
    devices = rng.choice(
        ["mobile", "desktop", "tablet"],
        size=n_travelers,
        p=[0.52, 0.38, 0.10],
    )

    # ── Booking channel ──
    channels = rng.choice(
        ["direct_web", "ota_app", "ota_web", "travel_agent", "metasearch"],
        size=n_travelers,
        p=[0.15, 0.30, 0.25, 0.15, 0.15],
    )

    # ── Loyalty & repeat ──
    loyalty_member = rng.choice(
        [True, False], size=n_travelers, p=[0.25, 0.75]
    )
    loyalty_member[purposes == "business"] = rng.choice(
        [True, False],
        size=(purposes == "business").sum(),
        p=[0.55, 0.45],
    )
    repeat_visitor = rng.choice(
        [True, False], size=n_travelers, p=[0.15, 0.85]
    )

    # ── Nationality region ──
    nationalities = rng.choice(
        ["US_Northeast", "US_Other", "International"],
        size=n_travelers,
        p=[0.55, 0.30, 0.15],
    )

    # ── Day of week preference for booking ──
    booking_dow = rng.choice(
        ["weekday", "weekend"], size=n_travelers, p=[0.62, 0.38]
    )
    booking_dow[purposes == "business"] = rng.choice(
        ["weekday", "weekend"],
        size=(purposes == "business").sum(),
        p=[0.82, 0.18],
    )

    # ── Assemble ──
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


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — CURATED & DECORRELATED
# ═══════════════════════════════════════════════════════════════

def engineer_clustering_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    
    df = df.copy()

    # ── Derived ratios (more informative than raw amounts) ──
    df["HOTEL_FLIGHT_RATIO"] = (
        df["TOTAL_HOTEL_SPEND"] / df["FLIGHT_SPEND"].clip(lower=1)
    ).round(3)

    df["SPEND_PER_NIGHT_PP"] = (
        df["HOTEL_SPEND_PER_NIGHT"] / df["GROUP_SIZE"]
    ).round(2)

    df["TOTAL_SPEND_PP"] = (
        df["TOTAL_TRIP_SPEND"] / df["GROUP_SIZE"]
    ).round(2)

    # ── Lead time: ordinal bucket instead of raw days ──
    df["LEAD_TIME_BUCKET"] = pd.cut(
        df["LEAD_TIME_DAYS"],
        bins=[0, 7, 21, 45, 90, 365],
        labels=[1, 2, 3, 4, 5],
    ).astype(float)

    # ── Binary encodings ──
    df["IS_LOYALTY"] = df["LOYALTY_MEMBER"].astype(int)
    df["IS_REPEAT"] = df["REPEAT_VISITOR"].astype(int)
    df["IS_MOBILE"] = (df["DEVICE"] == "mobile").astype(int)
    df["IS_DUBAI_PREF"] = (df["PREFERRED_MARKET"] == "Dubai").astype(int)

    # ── Price tier numeric ──
    tier_num = {"Budget": 1, "Mid-Range": 2, "Upscale": 3, "Luxury": 4}
    df["PRICE_TIER_NUM"] = df["PREFERRED_PRICE_TIER"].map(tier_num).fillna(2)

    cluster_features = [
        "AGE",
        "GROUP_SIZE",
        "STAY_NIGHTS",
        "LEAD_TIME_BUCKET",
        "SPEND_PER_NIGHT_PP",
        "HOTEL_FLIGHT_RATIO",
        "PRICE_TIER_NUM",
        "IS_LOYALTY",
        "IS_DUBAI_PREF",
    ]

    print(f"Engineered {len(cluster_features)} curated clustering features")
    return df, cluster_features


# ═══════════════════════════════════════════════════════════════
# K-MEANS — OPTIMAL K SEARCH WITH PCA
# ═══════════════════════════════════════════════════════════════

def find_optimal_k(
    df: pd.DataFrame,
    features: list[str],
    k_range: range = range(2, 9),
    seed: int = None,
    use_pca: bool = True,
    pca_variance: float = 0.90,
) -> pd.DataFrame:
    """
    Evaluate multiple k values using inertia, silhouette, and
    Calinski-Harabasz index.

    Parameters
    ----------
    df : DataFrame with clustering features
    features : list of column names to cluster on
    k_range : range of k values to evaluate
    seed : random seed for reproducibility
    use_pca : if True, apply PCA before clustering (recommended)
    pca_variance : fraction of variance to retain (0.90 = 90%)

    Returns DataFrame with one row per k and quality metrics.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required for clustering")

    seed = seed or AB_TEST_SEED
    X = df[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if use_pca:
        pca = PCA(n_components=pca_variance, random_state=seed)
        X_scaled = pca.fit_transform(X_scaled)
        n_comp = X_scaled.shape[1]
        explained = sum(pca.explained_variance_ratio_) * 100
        print(f"  PCA: {len(features)} features → {n_comp} components "
              f"({explained:.1f}% variance retained)")

    results = []
    for k in k_range:
        km = KMeans(
            n_clusters=k, random_state=seed, n_init=15, max_iter=500
        )
        labels = km.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, labels) if k > 1 else 0
        ch = calinski_harabasz_score(X_scaled, labels) if k > 1 else 0

        results.append({
            "K": k,
            "INERTIA": round(km.inertia_, 1),
            "SILHOUETTE": round(sil, 4),
            "CALINSKI_HARABASZ": round(ch, 1),
        })
        print(f"  k={k}: inertia={km.inertia_:.0f}, "
              f"silhouette={sil:.3f}, CH={ch:.0f}")

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# K-MEANS — RUN WITH CHOSEN K
# ═══════════════════════════════════════════════════════════════

def run_kmeans(
    df: pd.DataFrame,
    features: list[str],
    k: int = 4,
    seed: int = None,
    use_pca: bool = True,
    pca_variance: float = 0.90,
) -> tuple[pd.DataFrame, object, object]:
    """
    Run K-Means with chosen k. Optionally applies PCA first.

    Returns
    -------
    df : DataFrame with CLUSTER labels (and PCA_1, PCA_2 if use_pca)
    km : fitted KMeans model
    scaler : fitted StandardScaler
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required for clustering")

    seed = seed or AB_TEST_SEED
    df = df.copy()

    X = df[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = None
    if use_pca:
        pca = PCA(n_components=pca_variance, random_state=seed)
        X_clustered = pca.fit_transform(X_scaled)
        print(f"  PCA: {X_scaled.shape[1]} → {X_clustered.shape[1]} components")
    else:
        X_clustered = X_scaled

    km = KMeans(
        n_clusters=k, random_state=seed, n_init=15, max_iter=500
    )
    df["CLUSTER"] = km.fit_predict(X_clustered)

    sil = silhouette_score(X_clustered, df["CLUSTER"].values)
    ch = calinski_harabasz_score(X_clustered, df["CLUSTER"].values)
    print(f"\nK-Means (k={k}): silhouette={sil:.3f}, CH={ch:.0f}")
    print(f"  Cluster sizes: "
          f"{dict(df['CLUSTER'].value_counts().sort_index())}")

    # Store PCA components for 2D visualization
    if use_pca and X_clustered.shape[1] >= 2:
        df["PCA_1"] = X_clustered[:, 0]
        df["PCA_2"] = X_clustered[:, 1]

    return df, km, scaler


# ═══════════════════════════════════════════════════════════════
# CLUSTER PROFILING
# ═══════════════════════════════════════════════════════════════

def profile_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a profile summary for each cluster.
    Returns a DataFrame with one row per cluster and key metrics.
    """
    profiles = []
    for cluster in sorted(df["CLUSTER"].unique()):
        cdf = df[df["CLUSTER"] == cluster]
        n = len(cdf)

        profiles.append({
            "CLUSTER": cluster,
            "SIZE": n,
            "PCT_OF_TOTAL": round(n / len(df) * 100, 1),

            # Demographics
            "AVG_AGE": round(cdf["AGE"].mean(), 1),
            "AVG_GROUP_SIZE": round(cdf["GROUP_SIZE"].mean(), 1),

            # Trip behavior
            "AVG_STAY_NIGHTS": round(cdf["STAY_NIGHTS"].mean(), 1),
            "AVG_LEAD_TIME": round(cdf["LEAD_TIME_DAYS"].mean(), 1),
            "TOP_PURPOSE": cdf["TRIP_PURPOSE"].mode().iloc[0],
            "PURPOSE_CONCENTRATION": round(
                cdf["TRIP_PURPOSE"].value_counts(normalize=True).iloc[0]
                * 100, 1
            ),

            # Spending
            "AVG_FLIGHT_SPEND": round(cdf["FLIGHT_SPEND"].mean(), 0),
            "AVG_HOTEL_PER_NIGHT": round(
                cdf["HOTEL_SPEND_PER_NIGHT"].mean(), 0
            ),
            "AVG_TOTAL_SPEND": round(cdf["TOTAL_TRIP_SPEND"].mean(), 0),
            "MEDIAN_TOTAL_SPEND": round(
                cdf["TOTAL_TRIP_SPEND"].median(), 0
            ),

            # Preferences
            "TOP_PRICE_TIER": cdf["PREFERRED_PRICE_TIER"].mode().iloc[0],
            "PCT_DUBAI_PREF": round(
                (cdf["PREFERRED_MARKET"] == "Dubai").mean() * 100, 1
            ),
            "TOP_CHANNEL": cdf["BOOKING_CHANNEL"].mode().iloc[0],
            "TOP_DEVICE": cdf["DEVICE"].mode().iloc[0],

            # Loyalty
            "PCT_LOYALTY": round(cdf["LOYALTY_MEMBER"].mean() * 100, 1),
            "PCT_REPEAT": round(cdf["REPEAT_VISITOR"].mean() * 100, 1),

            # Fare class
            "TOP_FARE_CLASS": cdf["FARE_CLASS"].mode().iloc[0],
            "PCT_BUSINESS_CLASS": round(
                (cdf["FARE_CLASS"] == "business").mean() * 100, 1
            ),
        })

    return pd.DataFrame(profiles)


def label_clusters(profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign business-meaningful archetype labels based on cluster profiles.

    Uses a rule-based approach matching on:
      top purpose, spend level, stay duration, fare class, group size
    """
    profile_df = profile_df.copy()
    labels = []

    for _, row in profile_df.iterrows():
        purpose = row["TOP_PURPOSE"]
        avg_spend = row["AVG_TOTAL_SPEND"]
        stay = row["AVG_STAY_NIGHTS"]
        fare_pct_biz = row["PCT_BUSINESS_CLASS"]
        price_tier = row["TOP_PRICE_TIER"]
        group_sz = row["AVG_GROUP_SIZE"]

        if purpose == "business" or fare_pct_biz > 40:
            if avg_spend > 8000:
                labels.append("💼 Premium Business")
            else:
                labels.append("💼 Corporate Traveler")
        elif purpose == "honeymoon" or (
            price_tier == "Luxury" and stay >= 5
        ):
            labels.append("💍 Honeymoon/Luxury")
        elif purpose == "family_vacation" or group_sz >= 3:
            labels.append("👨‍👩‍👧‍👦 Family Explorer")
        elif purpose == "transit" or stay <= 2:
            labels.append("✈️ Transit/Stopover")
        elif purpose == "leisure" and avg_spend < 3000:
            labels.append("🎒 Budget Explorer")
        elif purpose == "leisure" and avg_spend >= 3000:
            labels.append("🌴 Comfort Leisure")
        else:
            labels.append(f"📦 Segment {row['CLUSTER']}")

    profile_df["ARCHETYPE"] = labels

    print("\nCluster Archetypes:")
    for _, row in profile_df.iterrows():
        print(f"  Cluster {row['CLUSTER']}: {row['ARCHETYPE']} "
              f"(n={row['SIZE']}, avg_spend=${row['AVG_TOTAL_SPEND']:,.0f})")

    return profile_df


# ═══════════════════════════════════════════════════════════════
# SEGMENT × MARKET CROSS-TABULATION
# ═══════════════════════════════════════════════════════════════

def segment_market_crosstab(
    df: pd.DataFrame,
    profile_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cross-tabulate segments with hotel market preference.
    Shows which segments skew Dubai vs NYC.
    """
    label_map = dict(zip(profile_df["CLUSTER"], profile_df["ARCHETYPE"]))
    df = df.copy()
    df["ARCHETYPE"] = df["CLUSTER"].map(label_map)

    ct = pd.crosstab(
        df["ARCHETYPE"], df["PREFERRED_MARKET"],
        margins=True, normalize="index",
    ).round(3) * 100

    ct.columns = [
        f"PCT_{c.upper()}" if c != "All" else "TOTAL_PCT"
        for c in ct.columns
    ]
    return ct


# ═══════════════════════════════════════════════════════════════
# SEGMENT-SPECIFIC OTA RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════

def generate_segment_recommendations(
    profile_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate OTA strategy recommendations per segment.
    """
    recs = []

    for _, row in profile_df.iterrows():
        archetype = row["ARCHETYPE"]

        if "Business" in archetype or "Corporate" in archetype:
            recs.append({
                "ARCHETYPE": archetype,
                "PRICING_STRATEGY": (
                    "Corporate rate cards, loyalty tier upgrades"
                ),
                "MARKETING_CHANNEL": (
                    "LinkedIn ads, email to frequent flyer lists"
                ),
                "PRODUCT_FOCUS": (
                    "Airport lounge access, flexible cancellation, "
                    "business hotel bundles"
                ),
                "UPSELL_OPP": (
                    "Business→First upgrade, late checkout, "
                    "meeting room packages"
                ),
                "RETENTION": (
                    "Priority loyalty tier, personalized trip reports"
                ),
            })
        elif "Honeymoon" in archetype or "Luxury" in archetype:
            recs.append({
                "ARCHETYPE": archetype,
                "PRICING_STRATEGY": (
                    "Premium bundles with experiences "
                    "(desert safari, spa)"
                ),
                "MARKETING_CHANNEL": (
                    "Instagram, wedding planning sites, "
                    "Google Display"
                ),
                "PRODUCT_FOCUS": (
                    "Romance packages, suite upgrades, "
                    "private transfers"
                ),
                "UPSELL_OPP": (
                    "Couples spa, sunset dinner cruise, "
                    "photography package"
                ),
                "RETENTION": (
                    "Anniversary reminder emails, "
                    "luxury brand partnerships"
                ),
            })
        elif "Family" in archetype:
            recs.append({
                "ARCHETYPE": archetype,
                "PRICING_STRATEGY": (
                    "Kids-stay-free bundles, family suite discounts"
                ),
                "MARKETING_CHANNEL": (
                    "Facebook, parenting blogs, YouTube family vlogs"
                ),
                "PRODUCT_FOCUS": (
                    "Family rooms, theme park tickets, "
                    "kid-friendly hotels"
                ),
                "UPSELL_OPP": (
                    "Waterpark access, babysitting services, "
                    "family photo shoots"
                ),
                "RETENTION": (
                    "School holiday deal alerts, "
                    "family loyalty program"
                ),
            })
        elif "Transit" in archetype or "Stopover" in archetype:
            recs.append({
                "ARCHETYPE": archetype,
                "PRICING_STRATEGY": (
                    "Ultra-competitive day rates, "
                    "<24h stay bundles"
                ),
                "MARKETING_CHANNEL": (
                    "In-app push during layover booking, "
                    "airline partner deals"
                ),
                "PRODUCT_FOCUS": (
                    "Airport hotels, transit visa included, "
                    "luggage storage"
                ),
                "UPSELL_OPP": (
                    "City tour add-on (4h Dubai highlights), "
                    "lounge access"
                ),
                "RETENTION": (
                    "Auto-suggest Dubai stopover on future "
                    "long-haul bookings"
                ),
            })
        elif "Budget" in archetype:
            recs.append({
                "ARCHETYPE": archetype,
                "PRICING_STRATEGY": (
                    "Flash sales, early-bird discounts, "
                    "price-match guarantee"
                ),
                "MARKETING_CHANNEL": (
                    "TikTok, Reddit r/travel, "
                    "Google Flights integration"
                ),
                "PRODUCT_FOCUS": (
                    "Budget hotels, hostels, "
                    "neighborhood guides for cheap eats"
                ),
                "UPSELL_OPP": (
                    "Travel insurance, SIM card bundle, "
                    "metro pass add-on"
                ),
                "RETENTION": (
                    "Price drop alerts, "
                    "deal of the week newsletter"
                ),
            })
        else:
            recs.append({
                "ARCHETYPE": archetype,
                "PRICING_STRATEGY": (
                    "Dynamic pricing based on lead time + demand"
                ),
                "MARKETING_CHANNEL": (
                    "Google Ads, metasearch, retargeting"
                ),
                "PRODUCT_FOCUS": (
                    "Curated mid-range hotel collections, "
                    "local experience add-ons"
                ),
                "UPSELL_OPP": (
                    "Hotel+flight bundle, airport transfer, "
                    "travel insurance"
                ),
                "RETENTION": (
                    "Post-trip review prompt, "
                    "seasonal deal emails"
                ),
            })

    return pd.DataFrame(recs)


# ═══════════════════════════════════════════════════════════════
# STATISTICAL VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_segments(
    df: pd.DataFrame,
    key_metrics: list[str] = None,
) -> pd.DataFrame:
    """
    Kruskal-Wallis tests to verify clusters are truly different
    on key behavioral dimensions.

    Non-parametric test — no normality assumption needed.
    """
    if key_metrics is None:
        key_metrics = [
            "TOTAL_TRIP_SPEND",
            "STAY_NIGHTS",
            "LEAD_TIME_DAYS",
            "FLIGHT_SPEND",
            "HOTEL_SPEND_PER_NIGHT",
            "GROUP_SIZE",
            "AGE",
        ]

    clusters = sorted(df["CLUSTER"].unique())
    groups = [df[df["CLUSTER"] == c] for c in clusters]

    results = []
    for metric in key_metrics:
        metric_groups = [g[metric].dropna().values for g in groups]

        if all(len(g) >= 5 for g in metric_groups):
            h_stat, p_val = stats.kruskal(*metric_groups)
        else:
            h_stat, p_val = np.nan, np.nan

        results.append({
            "METRIC": metric,
            "H_STATISTIC": (
                round(h_stat, 2) if not np.isnan(h_stat) else np.nan
            ),
            "P_VALUE": (
                round(p_val, 6) if not np.isnan(p_val) else np.nan
            ),
            "SIGNIFICANT": (
                p_val < 0.05 if not np.isnan(p_val) else False
            ),
        })

    val_df = pd.DataFrame(results)
    sig_count = val_df["SIGNIFICANT"].sum()
    print(f"\nSegment validation: {sig_count}/{len(val_df)} metrics "
          f"show significant inter-cluster differences")
    return val_df


# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_full_segmentation(
    hotel_df: pd.DataFrame = None,
    k: int = 4,
    use_pca: bool = True,
) -> dict:
    """
    Run the complete traveler segmentation pipeline.

    Parameters
    ----------
    hotel_df : prepared hotel data (from prepare_funnel_data)
    k : number of clusters (pass None to auto-select via silhouette)
    use_pca : whether to apply PCA before clustering

    Returns dict with all pipeline outputs.
    """
    print("\n" + "=" * 60)
    print("  M04: TRAVELER SEGMENTATION")
    print("=" * 60)

    # 1. Generate traveler profiles
    travelers = generate_traveler_profiles(hotel_df, n_travelers=2000)

    # 2. Engineer features
    travelers, features = engineer_clustering_features(travelers)

    # 3. Find optimal k
    print("\nSearching for optimal k...")
    k_eval = find_optimal_k(
        travelers, features, use_pca=use_pca
    )

    # Auto-select k if not provided
    if k is None:
        k = int(k_eval.loc[k_eval["SILHOUETTE"].idxmax(), "K"])
        print(f"\n🎯 Auto-selected k={k} (best silhouette)")

    # 4. Run K-Means
    travelers, km_model, scaler = run_kmeans(
        travelers, features, k=k, use_pca=use_pca
    )

    # 5. Profile clusters
    profiles = profile_clusters(travelers)
    profiles = label_clusters(profiles)

    # 6. Validate
    validation = validate_segments(travelers)

    # 7. Market cross-tab
    market_ct = segment_market_crosstab(travelers, profiles)

    # 8. Recommendations
    recommendations = generate_segment_recommendations(profiles)

    return {
        "travelers": travelers,
        "features": features,
        "k_evaluation": k_eval,
        "model": km_model,
        "scaler": scaler,
        "profiles": profiles,
        "validation": validation,
        "market_crosstab": market_ct,
        "recommendations": recommendations,
    }


# ═══════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.data_collection.google_places import generate_synthetic_hotels
    from src.analysis.funnel_analyzer import prepare_funnel_data

    hotels = generate_synthetic_hotels()
    hotels = prepare_funnel_data(hotels)

    results = run_full_segmentation(hotels, k=None, use_pca=True)

    print("\n" + "=" * 60)
    print("  PROFILES")
    print("=" * 60)
    print(results["profiles"][
        ["CLUSTER", "ARCHETYPE", "SIZE", "AVG_TOTAL_SPEND",
         "TOP_PURPOSE", "TOP_PRICE_TIER", "PCT_LOYALTY"]
    ].to_string(index=False))

    print("\n" + "=" * 60)
    print("  VALIDATION")
    print("=" * 60)
    print(results["validation"].to_string(index=False))

    print("\n✅ Segmentation pipeline complete")