"""
Aviation / Capacity Preprocessing
===================================
Prepare capacity + trends data into Prophet-ready format.
"""

import pandas as pd
from typing import Optional


def prepare_forecast_data(
    capacity_df: pd.DataFrame,
    trends_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge Aviation Edge capacity (supply) with Google Trends (demand signals)
    into a single monthly DataFrame for Prophet.

    Parameters
    ----------
    capacity_df : monthly capacity from Aviation Edge (EST_PASSENGERS, MONTHLY_FLIGHTS)
    trends_df : weekly Google Trends (optional — resampled to monthly)

    Returns
    -------
    Prophet-formatted DataFrame: 'ds', 'y', + regressor columns
    """
    prophet_df = capacity_df[["DATE", "EST_PASSENGERS"]].copy()
    prophet_df = prophet_df.rename(columns={"DATE": "ds", "EST_PASSENGERS": "y"})
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    if "MONTHLY_FLIGHTS" in capacity_df.columns:
        prophet_df["supply_flights"] = capacity_df["MONTHLY_FLIGHTS"].values
    if "LOAD_FACTOR" in capacity_df.columns:
        prophet_df["load_factor"] = capacity_df["LOAD_FACTOR"].values

    if trends_df is not None and not trends_df.empty:
        monthly_trends = trends_df.resample("MS").mean().round(1)
        monthly_trends = monthly_trends.reset_index()
        monthly_trends = monthly_trends.rename(columns={monthly_trends.columns[0]: "ds"})

        rename = {}
        for col in monthly_trends.columns[1:]:
            clean = f"trend_{col.lower().replace(' ', '_').replace('-', '_')}"
            rename[col] = clean
        monthly_trends = monthly_trends.rename(columns=rename)

        prophet_df = prophet_df.merge(monthly_trends, on="ds", how="left")

        regressor_cols = [c for c in prophet_df.columns if c.startswith("trend_")]
        prophet_df[regressor_cols] = prophet_df[regressor_cols].ffill().bfill()
        print(f"Added {len(regressor_cols)} Trends regressors: {regressor_cols}")

    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)
    print(f"Forecast data: {len(prophet_df)} months")
    return prophet_df
