from __future__ import annotations

import numpy as np
import pandas as pd


def build_policy_timeline(policy_events: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(policy_events).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def add_policy_regimes(ts_df: pd.DataFrame, policy_df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = ts_df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    pol = policy_df.copy().sort_values("date").reset_index(drop=True)

    out["policy_event"] = None
    out["friction_score"] = np.nan
    out["policy_direction"] = None

    idx = 0
    current = None
    for i, row in out.iterrows():
        while idx < len(pol) and pol.loc[idx, "date"] <= row[date_col]:
            current = pol.loc[idx]
            idx += 1
        if current is not None:
            out.at[i, "policy_event"] = current.get("event")
            out.at[i, "friction_score"] = current.get("friction_score")
            out.at[i, "policy_direction"] = current.get("direction")

    return out


def pre_post_impact(
    ts_df: pd.DataFrame,
    event_date: str,
    metric_col: str,
    window_days: int = 60,
    date_col: str = "date",
) -> dict:
    df = ts_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    d0 = pd.to_datetime(event_date)
    pre = df[(df[date_col] < d0) & (df[date_col] >= d0 - pd.Timedelta(days=window_days))]
    post = df[(df[date_col] >= d0) & (df[date_col] <= d0 + pd.Timedelta(days=window_days))]

    pre_mean = float(pre[metric_col].mean()) if len(pre) else np.nan
    post_mean = float(post[metric_col].mean()) if len(post) else np.nan

    if np.isnan(pre_mean) or pre_mean == 0 or np.isnan(post_mean):
        pct_change = np.nan
    else:
        pct_change = (post_mean - pre_mean) / pre_mean * 100

    return {
        "event_date": d0.date().isoformat(),
        "metric": metric_col,
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "pct_change": pct_change,
        "n_pre": len(pre),
        "n_post": len(post),
    }


def compute_composite_demand_index(df: pd.DataFrame, trend_col: str, flights_col: str) -> pd.DataFrame:
    out = df.copy()
    for c in [trend_col, flights_col]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    t_std = out[trend_col].std(ddof=0)
    f_std = out[flights_col].std(ddof=0)

    out["trend_z"] = (out[trend_col] - out[trend_col].mean()) / (t_std if t_std and t_std > 0 else 1.0)
    out["flights_z"] = (out[flights_col] - out[flights_col].mean()) / (f_std if f_std and f_std > 0 else 1.0)

    out["demand_index"] = 0.6 * out["trend_z"] + 0.4 * out["flights_z"]
    return out