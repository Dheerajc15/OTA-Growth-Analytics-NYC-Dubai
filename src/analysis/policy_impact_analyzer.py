"""
Module 06 — Policy Impact & Visa Friction Analyzer
====================================================
Data Sources: Google Trends (#3) + Aviation Edge (#4) + VISA_POLICY_EVENTS

Provides functions to:
  - Build policy timelines and annotate time-series with policy regimes
  - Compute composite demand index (trends + flight capacity)
  - Measure pre/post impact of policy shocks
  - Estimate demand recovery time after disruptions
  - Correlate friction scores with demand signals
  - Classify shock severity and generate OTA disruption playbooks
"""

import pandas as pd
import numpy as np
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# POLICY TIMELINE
# ═══════════════════════════════════════════════════════════════

def build_policy_timeline(policy_events: list[dict]) -> pd.DataFrame:
    """Convert VISA_POLICY_EVENTS list to sorted DataFrame."""
    df = pd.DataFrame(policy_events).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_policy_regimes(
    ts_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Annotate a time-series with the most-recent policy event,
    friction score, and policy direction at each row.
    """
    out = ts_df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)

    policy_df = policy_df.sort_values("date").reset_index(drop=True)
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


# ═══════════════════════════════════════════════════════════════
# COMPOSITE DEMAND INDEX
# ═══════════════════════════════════════════════════════════════

def compute_composite_demand_index(
    df: pd.DataFrame,
    trend_col: str,
    flights_col: str,
    trend_weight: float = 0.6,
    flights_weight: float = 0.4,
) -> pd.DataFrame:
    """
    Z-score normalise trends and flights, then combine with configurable weights.
    Default: 60 % trends + 40 % flights.
    """
    out = df.copy()
    out["trend_z"] = (out[trend_col] - out[trend_col].mean()) / (out[trend_col].std(ddof=0) + 1e-9)
    out["flights_z"] = (out[flights_col] - out[flights_col].mean()) / (out[flights_col].std(ddof=0) + 1e-9)
    out["demand_index"] = trend_weight * out["trend_z"] + flights_weight * out["flights_z"]
    return out


# ═══════════════════════════════════════════════════════════════
# PRE / POST IMPACT
# ═══════════════════════════════════════════════════════════════

def pre_post_impact(
    ts_df: pd.DataFrame,
    event_date: str,
    metric_col: str,
    window_days: int = 60,
    date_col: str = "date",
) -> dict:
    """Compute before/after means and percent-change around an event."""
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
        "pre_mean": round(pre_mean, 3) if not np.isnan(pre_mean) else np.nan,
        "post_mean": round(post_mean, 3) if not np.isnan(post_mean) else np.nan,
        "pct_change": round(pct_change, 2) if not np.isnan(pct_change) else np.nan,
        "n_pre": len(pre),
        "n_post": len(post),
    }


# ═══════════════════════════════════════════════════════════════
# RECOVERY TIME ESTIMATION
# ═══════════════════════════════════════════════════════════════

def estimate_recovery_time(
    ts_df: pd.DataFrame,
    event_date: str,
    metric_col: str,
    baseline_window_days: int = 90,
    max_search_days: int = 730,
    threshold_pct: float = 0.90,
    date_col: str = "date",
) -> dict:
    """
    After a negative shock, how many months until the metric recovers
    to *threshold_pct* of the pre-event baseline?

    Returns dict with recovery_days, recovery_months, and baseline info.
    """
    df = ts_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    d0 = pd.to_datetime(event_date)

    pre = df[(df[date_col] < d0) & (df[date_col] >= d0 - pd.Timedelta(days=baseline_window_days))]
    baseline = pre[metric_col].mean()
    target = baseline * threshold_pct

    post = df[df[date_col] >= d0].sort_values(date_col)
    if post.empty or np.isnan(baseline):
        return {"event_date": event_date, "recovery_days": None, "recovery_months": None,
                "baseline": baseline, "target": target}

    # Use a 3-period rolling mean to smooth noise
    post = post.copy()
    post["_smooth"] = post[metric_col].rolling(3, min_periods=1).mean()

    recovered = post[post["_smooth"] >= target]
    if recovered.empty:
        recovery_days = None
        recovery_months = None
    else:
        first_recovery = recovered.iloc[0][date_col]
        recovery_days = (first_recovery - d0).days
        recovery_months = round(recovery_days / 30.44, 1)

    return {
        "event_date": event_date,
        "baseline": round(baseline, 2),
        "target_90pct": round(target, 2),
        "recovery_days": recovery_days,
        "recovery_months": recovery_months,
    }


# ═══════════════════════════════════════════════════════════════
# FRICTION ↔ DEMAND CORRELATION
# ═══════════════════════════════════════════════════════════════

def friction_demand_correlation(
    df: pd.DataFrame,
    friction_col: str = "friction_score",
    demand_col: str = "demand_index",
) -> dict:
    """Pearson and Spearman correlation between friction and demand."""
    valid = df[[friction_col, demand_col]].dropna()
    if len(valid) < 5:
        return {"pearson": np.nan, "spearman": np.nan, "n": len(valid)}

    pearson = valid[friction_col].corr(valid[demand_col])
    spearman = valid[friction_col].corr(valid[demand_col], method="spearman")
    return {
        "pearson": round(pearson, 4),
        "spearman": round(spearman, 4),
        "n": len(valid),
    }


# ═══════════════════════════════════════════════════════════════
# REGIME-LEVEL SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════

def regime_summary_stats(
    df: pd.DataFrame,
    demand_col: str = "demand_index",
    regime_col: str = "policy_event",
) -> pd.DataFrame:
    """
    For each policy regime, compute mean/std/min/max of the demand index,
    plus the row count (months).
    """
    grouped = (
        df.groupby(regime_col)[demand_col]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={
            regime_col: "regime",
            "mean": "demand_mean",
            "std": "demand_std",
            "min": "demand_min",
            "max": "demand_max",
            "count": "months",
        })
    )
    grouped = grouped.sort_values("demand_mean", ascending=False).reset_index(drop=True)
    return grouped


# ═══════════════════════════════════════════════════════════════
# SHOCK CLASSIFIER
# ═══════════════════════════════════════════════════════════════

def classify_shock_severity(pct_change: float) -> str:
    """Classify demand change into severity bucket."""
    if pct_change is None or np.isnan(pct_change):
        return "unknown"
    if pct_change <= -50:
        return "catastrophic"
    if pct_change <= -25:
        return "severe"
    if pct_change <= -10:
        return "moderate"
    if pct_change < 0:
        return "mild"
    if pct_change < 10:
        return "neutral"
    if pct_change < 25:
        return "positive"
    return "strong_positive"


# ═══════════════════════════════════════════════════════════════
# MONTH-OVER-MONTH DEMAND MOMENTUM
# ═══════════════════════════════════════════════════════════════

def compute_demand_momentum(
    df: pd.DataFrame,
    demand_col: str = "demand_index",
    date_col: str = "date",
) -> pd.DataFrame:
    """Add month-over-month absolute and percentage change columns."""
    out = df.sort_values(date_col).copy()
    out["demand_mom"] = out[demand_col].diff()
    out["demand_mom_pct"] = out[demand_col].pct_change() * 100
    return out


# ═══════════════════════════════════════════════════════════════
# VISA-KEYWORD SEARCH SPIKE ALIGNMENT
# ═══════════════════════════════════════════════════════════════

def align_visa_spikes_to_events(
    trends_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    keyword: str = "Dubai visa",
    spike_std: float = 1.5,
    proximity_days: int = 30,
) -> pd.DataFrame:
    """
    Detect spikes in the visa keyword and check which spikes
    fall within *proximity_days* of a policy event.
    """
    if keyword not in trends_df.columns:
        raise ValueError(f"'{keyword}' not in trends columns")

    series = trends_df[keyword]
    rmean = series.rolling(12, min_periods=4).mean()
    rstd = series.rolling(12, min_periods=4).std()
    mask = series > (rmean + spike_std * rstd)
    spikes = trends_df[mask].copy()
    spikes["spike_date"] = spikes.index

    if spikes.empty:
        return pd.DataFrame()

    records = []
    for _, spike_row in spikes.iterrows():
        sd = spike_row["spike_date"]
        for _, ev in policy_df.iterrows():
            if abs((sd - ev["date"]).days) <= proximity_days:
                records.append({
                    "spike_date": sd,
                    "spike_value": spike_row[keyword],
                    "event_date": ev["date"],
                    "event": ev["event"],
                    "days_offset": (sd - ev["date"]).days,
                })

    return pd.DataFrame(records) if records else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# OTA DISRUPTION PLAYBOOK GENERATOR
# ═══════════════════════════════════════════════════════════════

_PLAYBOOK_RULES = {
    "catastrophic": {
        "pricing": "Pause dynamic pricing; freeze package bundles at pre-shock levels.",
        "marketing": "Switch ad spend to empathy messaging and 'future travel credit' campaigns.",
        "inventory": "Halt new inventory commits; negotiate force-majeure clauses with hotels.",
        "crm": "Auto-send refund/reschedule emails within 24 h; waive change fees.",
    },
    "severe": {
        "pricing": "Reduce minimum margins by 30 %; introduce flexible-date bundles.",
        "marketing": "Shift budget to 'plan ahead' and visa-assistance content marketing.",
        "inventory": "Request 60-day cancellation windows from suppliers.",
        "crm": "Proactive outreach to high-LTV customers with flexible rebooking.",
    },
    "moderate": {
        "pricing": "Add 5-8 % discount codes for affected departure dates.",
        "marketing": "Boost retargeting for users who searched but didn't book.",
        "inventory": "Monitor load factors weekly; reduce exposure if LF < 70 %.",
        "crm": "Targeted push notifications about eased restrictions.",
    },
    "mild": {
        "pricing": "No change; monitor weekly.",
        "marketing": "Maintain BAU media mix; test messaging about smooth entry process.",
        "inventory": "Business as usual.",
        "crm": "Segment-specific nudges for visa-anxious browsers.",
    },
    "neutral": {
        "pricing": "Continue standard dynamic pricing.",
        "marketing": "A/B test seasonal creative.",
        "inventory": "Standard allocation.",
        "crm": "Loyalty program engagement campaigns.",
    },
    "positive": {
        "pricing": "Test 3-5 % price lift on premium bundles.",
        "marketing": "Amplify 'hassle-free entry' messaging across channels.",
        "inventory": "Increase forward inventory commits by 10 %.",
        "crm": "Win-back dormant users with 'travel is back' emails.",
    },
    "strong_positive": {
        "pricing": "Launch premium visa-inclusive bundles at higher AOV.",
        "marketing": "Full-funnel blitz: awareness -> consideration -> conversion.",
        "inventory": "Lock in bulk hotel allotments at current rates.",
        "crm": "VIP early-access deals for top-tier loyalty members.",
    },
}


def generate_disruption_playbook(impact_rows: list[dict]) -> pd.DataFrame:
    """
    Given a list of pre_post_impact dicts, classify each shock
    and attach the recommended OTA playbook.

    Returns a DataFrame with columns:
      event_date, pct_change, severity, pricing, marketing, inventory, crm
    """
    records = []
    for row in impact_rows:
        sev = classify_shock_severity(row.get("pct_change"))
        rules = _PLAYBOOK_RULES.get(sev, _PLAYBOOK_RULES["neutral"])
        records.append({
            "event_date": row["event_date"],
            "metric": row.get("metric", ""),
            "pct_change": row.get("pct_change"),
            "severity": sev,
            **rules,
        })
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# YEAR-OVER-YEAR COMPARISON AROUND EVENTS
# ═══════════════════════════════════════════════════════════════

def yoy_comparison(
    df: pd.DataFrame,
    demand_col: str = "demand_index",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    For each month, compute year-over-year change in demand.
    Useful for isolating shock effects from seasonal patterns.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month

    pivot = out.pivot_table(index="month", columns="year", values=demand_col, aggfunc="mean")
    yoy = pivot.pct_change(axis=1) * 100
    return yoy.round(2)
