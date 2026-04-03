import pandas as pd
import numpy as np

def build_policy_timeline(policy_events: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(policy_events).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def add_policy_regimes(ts_df: pd.DataFrame, policy_df: pd.DataFrame, date_col="date") -> pd.DataFrame:
    out = ts_df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col)

    policy_df = policy_df.sort_values("date")
    out["policy_event"] = None
    out["friction_score"] = np.nan
    out["policy_direction"] = None

    idx = 0
    current = None
    for i, row in out.iterrows():
        while idx < len(policy_df) and policy_df.loc[idx, "date"] <= row[date_col]:
            current = policy_df.loc[idx]
            idx += 1
        if current is not None:
            out.at[i, "policy_event"] = current["event"]
            out.at[i, "friction_score"] = current["friction_score"]
            out.at[i, "policy_direction"] = current["direction"]

    return out

def pre_post_impact(ts_df: pd.DataFrame, event_date: str, metric_col: str, window_days: int = 60, date_col="date"):
    df = ts_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    d0 = pd.to_datetime(event_date)

    pre = df[(df[date_col] < d0) & (df[date_col] >= d0 - pd.Timedelta(days=window_days))]
    post = df[(df[date_col] >= d0) & (df[date_col] <= d0 + pd.Timedelta(days=window_days))]

    pre_mean = pre[metric_col].mean()
    post_mean = post[metric_col].mean()
    pct_change = ((post_mean - pre_mean) / pre_mean * 100) if pre_mean and not np.isnan(pre_mean) else np.nan

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
    out["trend_z"] = (out[trend_col] - out[trend_col].mean()) / (out[trend_col].std(ddof=0) + 1e-9)
    out["flights_z"] = (out[flights_col] - out[flights_col].mean()) / (out[flights_col].std(ddof=0) + 1e-9)
    out["demand_index"] = 0.6 * out["trend_z"] + 0.4 * out["flights_z"]
    return out