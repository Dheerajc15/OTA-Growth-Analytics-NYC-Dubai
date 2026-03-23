"""
BTS DB1B Data Loader & Cleaner
================================
Bureau of Transportation Statistics — DB1B Coupon dataset.
Contains actual origin-destination passenger volumes and fares for US air routes.

Download manually from:
    https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoession_ID=0&Table_ID=272

Instructions:
    1. Table: DB1B Coupon
    2. Select fields: Year, Quarter, Origin, Dest, Passengers, MktFare, MktDistance
    3. Filter: Year >= 2018
    4. Download CSV(s), save to: data/raw/bts/
    
    NOTE: BTS does not let you filter by airport in the download UI for DB1B.
    Download the full dataset and this script filters to JFK/EWR/LGA → DXB.
    If full dataset is too large, download year-by-year.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Use try/except so the module can be imported even if config isn't set up yet
try:
    from config.settings import DATA_RAW, ORIGIN_AIRPORTS, DESTINATION_AIRPORT
except ImportError:
    DATA_RAW = Path("data/raw")
    ORIGIN_AIRPORTS = ["JFK", "EWR", "LGA"]
    DESTINATION_AIRPORT = "DXB"


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_bts_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load BTS DB1B Coupon CSV data.
    
    Parameters
    ----------
    filepath : Path, optional
        Path to a specific CSV file. If None, loads all CSVs in data/raw/bts/
    
    Returns
    -------
    pd.DataFrame
        Raw BTS data with all columns as downloaded.
    
    Raises
    ------
    FileNotFoundError
        If no CSV files are found in the expected directory.
    
    Example
    -------
    >>> df = load_bts_data()
    >>> df.shape
    (150000, 8)
    """
    bts_dir = DATA_RAW / "bts"
    
    if filepath:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        print(f"Loading BTS data from: {filepath}")
        return pd.read_csv(filepath, low_memory=False)
    
    # Load all CSVs from the bts directory
    csv_files = sorted(bts_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"No CSV files found in: {bts_dir}\n\n"
            f"To get the data:\n"
            f"1. Go to: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoession_ID=0&Table_ID=272\n"
            f"2. Select table: DB1B Coupon\n"
            f"3. Choose fields: Year, Quarter, Origin, Dest, Passengers, MktFare, MktDistance\n"
            f"4. Download and save CSV files to: {bts_dir}/\n"
            f"{'='*60}"
        )
    
    print(f"Found {len(csv_files)} BTS CSV file(s):")
    for f in csv_files:
        print(f"  → {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Concatenate all files
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
            print(f"  ✓ Loaded {f.name}: {len(df):,} rows")
        except Exception as e:
            print(f"  ✗ Error loading {f.name}: {e}")
    
    if not dfs:
        raise ValueError("All CSV files failed to load.")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined):,} total rows, {len(combined.columns)} columns")
    return combined


# ═══════════════════════════════════════════════════════════════
# DATA CLEANING
# ═══════════════════════════════════════════════════════════════

def clean_bts_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw BTS data and filter to NYC → Dubai route.
    
    Processing steps:
        1. Standardize column names to UPPERCASE
        2. Filter to ORIGIN in (JFK, EWR, LGA) and DEST = DXB
        3. Remove rows with missing passengers or fares
        4. Create date column from Year + Quarter
        5. Aggregate to quarterly level per origin airport
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw BTS DataFrame from load_bts_data()
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:
        DATE, YEAR, QUARTER, ORIGIN, TOTAL_PASSENGERS, AVG_FARE, 
        MEDIAN_FARE, TOTAL_ITINERARIES
    """
    # Step 1: Standardize column names
    df = df.copy()
    df.columns = df.columns.str.strip().str.upper()
    
    print(f"Available columns: {list(df.columns)}")
    
    # Step 2: Filter to NYC → DXB route
    # Check which airport columns exist (BTS uses different names across tables)
    origin_col = "ORIGIN" if "ORIGIN" in df.columns else None
    dest_col = "DEST" if "DEST" in df.columns else None
    
    if origin_col is None:
        # Try alternate column names
        for col in ["ORIGIN_AIRPORT_ID", "ORIGINAIRPORTID", "ORIGIN_CITY_NAME"]:
            if col in df.columns:
                origin_col = col
                break
    
    if dest_col is None:
        for col in ["DEST_AIRPORT_ID", "DESTAIRPORTID", "DEST_CITY_NAME"]:
            if col in df.columns:
                dest_col = col
                break
    
    if origin_col is None or dest_col is None:
        raise ValueError(
            f"Could not find origin/dest columns. Available: {list(df.columns)}"
        )
    
    print(f"Using origin column: '{origin_col}', dest column: '{dest_col}'")
    
    # Filter to our route
    mask_origin = df[origin_col].isin(ORIGIN_AIRPORTS)
    mask_dest = df[dest_col] == DESTINATION_AIRPORT
    route_df = df[mask_origin & mask_dest].copy()
    
    print(f"Filtered: {len(df):,} → {len(route_df):,} rows (NYC → DXB)")
    
    if len(route_df) == 0:
        print("\n⚠️  WARNING: No JFK/EWR/LGA → DXB records found!")
        print("This could mean:")
        print("  - The CSV doesn't contain international routes")
        print("  - Column values use airport IDs instead of codes")
        print(f"  - Sample origin values: {df[origin_col].unique()[:10]}")
        print(f"  - Sample dest values: {df[dest_col].unique()[:10]}")
        return pd.DataFrame()
    
    # Step 3: Clean numeric columns
    passenger_col = "PASSENGERS" if "PASSENGERS" in route_df.columns else None
    fare_col = "MKTFARE" if "MKTFARE" in route_df.columns else None
    
    if passenger_col is None:
        for col in route_df.columns:
            if "PASSENGER" in col:
                passenger_col = col
                break
    
    if fare_col is None:
        for col in route_df.columns:
            if "FARE" in col:
                fare_col = col
                break
    
    print(f"Using passenger column: '{passenger_col}', fare column: '{fare_col}'")
    
    # Convert to numeric and drop invalids
    if passenger_col:
        route_df[passenger_col] = pd.to_numeric(route_df[passenger_col], errors="coerce")
        route_df = route_df[route_df[passenger_col] > 0]
    
    if fare_col:
        route_df[fare_col] = pd.to_numeric(route_df[fare_col], errors="coerce")
        route_df = route_df[route_df[fare_col] > 0]
    
    # Step 4: Create date column from YEAR + QUARTER
    quarter_to_month = {1: 1, 2: 4, 3: 7, 4: 10}
    route_df["DATE"] = pd.to_datetime(
        route_df["YEAR"].astype(int).astype(str) + "-" +
        route_df["QUARTER"].map(quarter_to_month).astype(str) + "-01"
    )
    
    # Step 5: Aggregate to quarterly per-airport level
    # Rename for consistency before aggregation
    rename_map = {}
    if origin_col != "ORIGIN":
        rename_map[origin_col] = "ORIGIN"
    if passenger_col and passenger_col != "PASSENGERS":
        rename_map[passenger_col] = "PASSENGERS"
    if fare_col and fare_col != "MKTFARE":
        rename_map[fare_col] = "MKTFARE"
    
    if rename_map:
        route_df = route_df.rename(columns=rename_map)
    
    agg_dict = {"PASSENGERS": "sum"}
    if "MKTFARE" in route_df.columns:
        agg_dict["MKTFARE"] = ["mean", "median"]
    
    monthly = (
        route_df.groupby(["DATE", "YEAR", "QUARTER", "ORIGIN"])
        .agg(
            TOTAL_PASSENGERS=("PASSENGERS", "sum"),
            AVG_FARE=("MKTFARE", "mean") if "MKTFARE" in route_df.columns else ("PASSENGERS", "count"),
            MEDIAN_FARE=("MKTFARE", "median") if "MKTFARE" in route_df.columns else ("PASSENGERS", "count"),
            TOTAL_ITINERARIES=("PASSENGERS", "count"),
        )
        .reset_index()
    )
    
    print(f"Aggregated: {len(monthly)} quarterly airport-level records")
    print(f"Date range: {monthly['DATE'].min()} → {monthly['DATE'].max()}")
    print(f"Airports: {monthly['ORIGIN'].unique().tolist()}")
    
    return monthly.sort_values("DATE").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# ROUTE-LEVEL SUMMARY
# ═══════════════════════════════════════════════════════════════

def get_route_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across all NYC airports for total route demand.
    
    Parameters
    ----------
    df : pd.DataFrame
        Output from clean_bts_data() — per-airport quarterly data.
    
    Returns
    -------
    pd.DataFrame
        One row per quarter with total passengers and avg fare across
        all NYC-area airports.
    """
    if df.empty:
        return df
    
    summary = (
        df.groupby(["DATE", "YEAR", "QUARTER"])
        .agg(
            TOTAL_PASSENGERS=("TOTAL_PASSENGERS", "sum"),
            AVG_FARE=("AVG_FARE", "mean"),
            MEDIAN_FARE=("MEDIAN_FARE", "mean"),
            TOTAL_ITINERARIES=("TOTAL_ITINERARIES", "sum"),
            NUM_AIRPORTS=("ORIGIN", "nunique"),
        )
        .reset_index()
    )
    
    # Add derived features useful for forecasting
    summary["PASSENGERS_PER_DAY"] = summary["TOTAL_PASSENGERS"] / 90  # ~90 days/quarter
    summary["YOY_GROWTH"] = summary["TOTAL_PASSENGERS"].pct_change(periods=4)  # vs same quarter last year
    summary["QOQ_GROWTH"] = summary["TOTAL_PASSENGERS"].pct_change(periods=1)  # vs previous quarter
    
    return summary.sort_values("DATE").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# QUICK STATS
# ═══════════════════════════════════════════════════════════════

def print_route_stats(df: pd.DataFrame) -> None:
    """Print a quick summary of the route data for sanity checking."""
    if df.empty:
        print("⚠️  No data to summarize.")
        return
    
    print("\n" + "=" * 60)
    print("  NYC → DUBAI ROUTE SUMMARY (BTS DB1B)")
    print("=" * 60)
    print(f"  Date range:        {df['DATE'].min().strftime('%Y-Q%q') if 'DATE' in df.columns else 'N/A'}")
    print(f"                  →  {df['DATE'].max().strftime('%Y-Q%q') if 'DATE' in df.columns else 'N/A'}")
    print(f"  Total quarters:    {len(df)}")
    
    if "TOTAL_PASSENGERS" in df.columns:
        print(f"  Total passengers:  {df['TOTAL_PASSENGERS'].sum():,.0f}")
        print(f"  Avg/quarter:       {df['TOTAL_PASSENGERS'].mean():,.0f}")
        print(f"  Peak quarter:      {df.loc[df['TOTAL_PASSENGERS'].idxmax(), 'DATE'].strftime('%Y-Q%q')} "
              f"({df['TOTAL_PASSENGERS'].max():,.0f} pax)")
        print(f"  Lowest quarter:    {df.loc[df['TOTAL_PASSENGERS'].idxmin(), 'DATE'].strftime('%Y-Q%q')} "
              f"({df['TOTAL_PASSENGERS'].min():,.0f} pax)")
    
    if "AVG_FARE" in df.columns:
        print(f"  Avg fare:          ${df['AVG_FARE'].mean():,.0f}")
        print(f"  Fare range:        ${df['AVG_FARE'].min():,.0f} – ${df['AVG_FARE'].max():,.0f}")
    
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR (for development/testing without BTS download)
# ═══════════════════════════════════════════════════════════════

def generate_synthetic_bts_data(
    start_year: int = 2018,
    end_year: int = 2025,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic BTS data for development and testing.
    
    Use this when you haven't downloaded the real BTS data yet.
    The synthetic data mimics real patterns:
      - Winter (Q4, Q1) peaks for Dubai tourism
      - Ramadan dips (varies by year, mostly Q2)
      - COVID collapse in 2020
      - Post-COVID recovery trend
      - JFK > EWR >> LGA traffic distribution
    
    Parameters
    ----------
    start_year : int
    end_year : int
    seed : int
    
    Returns
    -------
    pd.DataFrame
        Synthetic data in same format as clean_bts_data() output.
    """
    np.random.seed(seed)
    
    records = []
    base_passengers = {
        "JFK": 5000,   # JFK is the primary Emirates hub
        "EWR": 2500,   # EWR has some traffic
        "LGA": 200,    # LGA has minimal international
    }
    
    # Seasonal multipliers by quarter
    seasonal = {
        1: 1.15,   # Jan-Mar: winter tourism to Dubai (peak)
        2: 0.80,   # Apr-Jun: Ramadan + getting hot in Dubai
        3: 0.65,   # Jul-Sep: extreme heat, low tourism
        4: 1.25,   # Oct-Dec: Dubai season starts, NYE spike
    }
    
    # Year-over-year growth trend (with COVID shock)
    yearly_factor = {
        2018: 1.00, 2019: 1.08, 2020: 0.25,  # COVID crash
        2021: 0.55, 2022: 0.90, 2023: 1.15,   # recovery
        2024: 1.25, 2025: 1.32,                 # growth
    }
    
    for year in range(start_year, end_year + 1):
        for quarter in [1, 2, 3, 4]:
            for airport, base_pax in base_passengers.items():
                # Apply factors
                pax = base_pax
                pax *= seasonal[quarter]
                pax *= yearly_factor.get(year, 1.0)
                
                # Add noise (±15%)
                pax *= np.random.uniform(0.85, 1.15)
                pax = max(int(pax), 0)
                
                # Fare: base ~$620 economy, varies by season and demand
                fare_base = 620
                fare = fare_base * (1 + (seasonal[quarter] - 1) * 0.5)  # higher fares in peak
                fare *= np.random.uniform(0.85, 1.15)
                
                # Create date
                quarter_to_month = {1: 1, 2: 4, 3: 7, 4: 10}
                date = pd.Timestamp(year=year, month=quarter_to_month[quarter], day=1)
                
                records.append({
                    "DATE": date,
                    "YEAR": year,
                    "QUARTER": quarter,
                    "ORIGIN": airport,
                    "TOTAL_PASSENGERS": pax,
                    "AVG_FARE": round(fare, 2),
                    "MEDIAN_FARE": round(fare * np.random.uniform(0.90, 0.98), 2),
                    "TOTAL_ITINERARIES": max(int(pax * np.random.uniform(0.3, 0.5)), 1),
                })
    
    df = pd.DataFrame(records)
    print(f"Generated {len(df)} synthetic BTS records ({start_year}–{end_year})")
    return df.sort_values("DATE").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# CLI ENTRYPOINT (run this file directly for quick test)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("BTS DB1B Loader — Quick Test")
    print("-" * 40)
    
    try:
        # Try loading real data first
        raw = load_bts_data()
        clean = clean_bts_data(raw)
        summary = get_route_summary(clean)
        print_route_stats(summary)
    except FileNotFoundError:
        print("\nNo real BTS data found. Generating synthetic data for testing...")
        synthetic = generate_synthetic_bts_data()
        summary = get_route_summary(synthetic)
        print_route_stats(summary)
        
        # Save synthetic data so notebooks can use it
        output_path = DATA_RAW / "bts" / "synthetic_bts_nyc_dxb.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(output_path, index=False)
        print(f"\nSaved synthetic data to: {output_path}")