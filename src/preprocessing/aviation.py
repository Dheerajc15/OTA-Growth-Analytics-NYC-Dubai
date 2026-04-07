from __future__ import annotations

import pandas as pd
from typing import Optional


def prepare_forecast_data(
    capacity_df: pd.DataFrame,
    trends_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    # Capacity -> base forecast frame
    c = capacity_df.copy()
    c["DATE"] = pd.to_datetime(c["DATE"], errors="coerce")
    c = c.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)

    out = c[["DATE", "EST_PASSENGERS"]].rename(columns={"DATE": "ds", "EST_PASSENGERS": "y"})
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")

    if "MONTHLY_FLIGHTS" in c.columns:
        out["supply_flights"] = pd.to_numeric(c["MONTHLY_FLIGHTS"], errors="coerce")
    if "LOAD_FACTOR" in c.columns:
        out["load_factor"] = pd.to_numeric(c["LOAD_FACTOR"], errors="coerce")

    # Optional trends regressors
    if trends_df is not None and not trends_df.empty:
        t = trends_df.copy()

        # Normalize trends time axis robustly
        if "DATE" in t.columns:
            t["DATE"] = pd.to_datetime(t["DATE"], errors="coerce")
            t = t.dropna(subset=["DATE"]).set_index("DATE")
        elif "date" in t.columns:
            t["date"] = pd.to_datetime(t["date"], errors="coerce")
            t = t.dropna(subset=["date"]).set_index("date")
        else:
            t.index = pd.to_datetime(t.index, errors="coerce")

        t = t.dropna(axis=0, how="all")
        t = t.select_dtypes(include=["number"])

        t_m = t.resample("MS").mean().reset_index()

        # Force first column to ds (works whether it is DATE/date/index name)
        t_m = t_m.rename(columns={t_m.columns[0]: "ds"})
        t_m["ds"] = pd.to_datetime(t_m["ds"], errors="coerce")

        # Prefix trend columns
        rename_map = {col: f"trend_{str(col).lower().replace(' ', '_').replace('-', '_')}"
                      for col in t_m.columns if col != "ds"}
        t_m = t_m.rename(columns=rename_map)

        out = out.merge(t_m, on="ds", how="left")

        reg_cols = [col for col in out.columns if col.startswith("trend_")]
        if reg_cols:
            out[reg_cols] = out[reg_cols].ffill().bfill()

    out = out.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    return out