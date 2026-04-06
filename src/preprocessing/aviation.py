from __future__ import annotations

import pandas as pd
from typing import Optional


def prepare_forecast_data(capacity_df: pd.DataFrame, trends_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    c = capacity_df.copy()
    c["DATE"] = pd.to_datetime(c["DATE"], errors="coerce")
    c = c.dropna(subset=["DATE"]).sort_values("DATE")

    out = c[["DATE", "EST_PASSENGERS"]].rename(columns={"DATE": "ds", "EST_PASSENGERS": "y"})
    out["y"] = pd.to_numeric(out["y"], errors="coerce")

    if "MONTHLY_FLIGHTS" in c.columns:
        out["supply_flights"] = pd.to_numeric(c["MONTHLY_FLIGHTS"], errors="coerce")
    if "LOAD_FACTOR" in c.columns:
        out["load_factor"] = pd.to_numeric(c["LOAD_FACTOR"], errors="coerce")

    if trends_df is not None and not trends_df.empty:
        t = trends_df.copy()
        if "DATE" in t.columns:
            t = t.set_index(pd.to_datetime(t["DATE"], errors="coerce")).drop(columns=["DATE"], errors="ignore")
        else:
            t.index = pd.to_datetime(t.index, errors="coerce")
        t = t.dropna(axis=0, how="all")
        t_m = t.resample("MS").mean().reset_index().rename(columns={"index": "ds"})
        t_m = t_m.rename(columns={cname: f"trend_{cname.lower().replace(' ', '_')}" for cname in t_m.columns if cname != "ds"})
        out = out.merge(t_m, on="ds", how="left")
        reg = [cname for cname in out.columns if cname.startswith("trend_")]
        out[reg] = out[reg].ffill().bfill()

    out = out.dropna(subset=["ds", "y"]).reset_index(drop=True)
    return out