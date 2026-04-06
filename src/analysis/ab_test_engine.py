"""
A/B Test Engine 
===========================
Robust utilities for simulated pricing experiments:
- conversion rate summaries
- z-test for proportions
- bootstrap confidence intervals
- segment-level diagnostics with minimum sample guards
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def summarize_ab(df: pd.DataFrame, group_col: str = "GROUP", outcome_col: str = "CONVERTED") -> pd.DataFrame:
    x = df.copy()
    x[outcome_col] = pd.to_numeric(x[outcome_col], errors="coerce").fillna(0).astype(int)

    summ = (
        x.groupby(group_col)
        .agg(
            N=(outcome_col, "size"),
            CONVERSIONS=(outcome_col, "sum"),
        )
        .reset_index()
    )
    summ["CVR"] = summ["CONVERSIONS"] / summ["N"].clip(lower=1)
    return summ


def ztest_proportions(
    n_control: int,
    x_control: int,
    n_treatment: int,
    x_treatment: int,
    alpha: float = 0.05,
) -> dict:
    p1 = x_control / max(n_control, 1)
    p2 = x_treatment / max(n_treatment, 1)
    diff = p2 - p1

    pooled = (x_control + x_treatment) / max((n_control + n_treatment), 1)
    se = np.sqrt(pooled * (1 - pooled) * (1 / max(n_control, 1) + 1 / max(n_treatment, 1)))

    if se <= 0:
        z = 0.0
        p = 1.0
    else:
        z = diff / se
        p = 2 * (1 - norm.cdf(abs(z)))

    zcrit = norm.ppf(1 - alpha / 2)
    ci_low = diff - zcrit * se
    ci_high = diff + zcrit * se

    return {
        "cvr_control": p1,
        "cvr_treatment": p2,
        "abs_diff": diff,
        "rel_lift_pct": (diff / p1 * 100) if p1 > 0 else np.nan,
        "z_stat": z,
        "p_value": p,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": p < alpha,
    }


def bootstrap_conversion_diff(
    df: pd.DataFrame,
    group_col: str = "GROUP",
    outcome_col: str = "CONVERTED",
    control_label: str = "control",
    treatment_label: str = "treatment",
    n_bootstrap: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    rng = np.random.RandomState(seed)
    x = df.copy()
    x[outcome_col] = pd.to_numeric(x[outcome_col], errors="coerce").fillna(0).astype(int)

    c = x[x[group_col] == control_label][outcome_col].values
    t = x[x[group_col] == treatment_label][outcome_col].values

    if len(c) == 0 or len(t) == 0:
        raise ValueError("Control or treatment group is empty.")

    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        c_s = rng.choice(c, size=len(c), replace=True)
        t_s = rng.choice(t, size=len(t), replace=True)
        diffs[i] = t_s.mean() - c_s.mean()

    mean_diff = float(diffs.mean())
    low = float(np.quantile(diffs, alpha / 2))
    high = float(np.quantile(diffs, 1 - alpha / 2))

    return {
        "bootstrap_diffs": diffs,
        "mean_diff": mean_diff,
        "ci_lower": low,
        "ci_upper": high,
        "significant": (low > 0) or (high < 0),
    }


def segment_ab_analysis(
    df: pd.DataFrame,
    segment_col: str,
    group_col: str = "GROUP",
    outcome_col: str = "CONVERTED",
    control_label: str = "control",
    treatment_label: str = "treatment",
    min_n_per_arm: int = 50,
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for seg, sdf in df.groupby(segment_col):
        c = sdf[sdf[group_col] == control_label]
        t = sdf[sdf[group_col] == treatment_label]

        n_c, n_t = len(c), len(t)
        x_c = int(pd.to_numeric(c[outcome_col], errors="coerce").fillna(0).sum())
        x_t = int(pd.to_numeric(t[outcome_col], errors="coerce").fillna(0).sum())

        if n_c < min_n_per_arm or n_t < min_n_per_arm:
            rows.append({
                "SEGMENT_VALUE": seg,
                "N_CONTROL": n_c,
                "N_TREATMENT": n_t,
                "CVR_CONTROL": x_c / max(n_c, 1),
                "CVR_TREATMENT": x_t / max(n_t, 1),
                "ABS_DIFF": np.nan,
                "RELATIVE_LIFT_PCT": np.nan,
                "P_VALUE": np.nan,
                "SIGNIFICANT": False,
                "SKIP_REASON": f"min_n_per_arm<{min_n_per_arm}",
            })
            continue

        res = ztest_proportions(n_c, x_c, n_t, x_t, alpha=alpha)
        rows.append({
            "SEGMENT_VALUE": seg,
            "N_CONTROL": n_c,
            "N_TREATMENT": n_t,
            "CVR_CONTROL": res["cvr_control"],
            "CVR_TREATMENT": res["cvr_treatment"],
            "ABS_DIFF": res["abs_diff"],
            "RELATIVE_LIFT_PCT": res["rel_lift_pct"],
            "P_VALUE": res["p_value"],
            "SIGNIFICANT": res["significant"],
            "SKIP_REASON": "",
        })

    out = pd.DataFrame(rows)
    return out.sort_values("RELATIVE_LIFT_PCT", ascending=False, na_position="last").reset_index(drop=True)


def sequential_monitoring(
    df: pd.DataFrame,
    time_col: str,
    group_col: str = "GROUP",
    outcome_col: str = "CONVERTED",
    checkpoints: int = 10,
) -> pd.DataFrame:
    """
    Monitoring-only utility.
    IMPORTANT: sequential p-values are not final fixed-horizon inference.
    """
    x = df.copy()
    x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
    x = x.dropna(subset=[time_col]).sort_values(time_col)

    n = len(x)
    if n == 0:
        return pd.DataFrame()

    cuts = np.linspace(int(n / checkpoints), n, checkpoints).astype(int)
    rows = []

    for k in cuts:
        s = x.iloc[:k]
        summ = summarize_ab(s, group_col=group_col, outcome_col=outcome_col)
        if len(summ) < 2:
            continue

        # assume rows contain control/treatment labels
        try:
            c = summ[summ[group_col].str.lower() == "control"].iloc[0]
            t = summ[summ[group_col].str.lower() == "treatment"].iloc[0]
        except Exception:
            # fallback to first two groups
            c, t = summ.iloc[0], summ.iloc[1]

        res = ztest_proportions(
            int(c["N"]), int(c["CONVERSIONS"]),
            int(t["N"]), int(t["CONVERSIONS"])
        )
        rows.append({
            "CHECKPOINT_N": k,
            "CVR_CONTROL": res["cvr_control"],
            "CVR_TREATMENT": res["cvr_treatment"],
            "ABS_DIFF": res["abs_diff"],
            "P_VALUE": res["p_value"],
            "SIGNIFICANT": res["significant"],
            "NOTE": "Monitoring only; adjust for sequential testing if making stop/go decisions.",
        })

    return pd.DataFrame(rows)