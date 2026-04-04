"""
A/B Test Engine — Bundle vs Hotel-Only Pricing (Module 03)
============================================================
Data Source: Simulated Fare Data (numpy) — Data Source #6

Business Question:
  "Does offering a flight+hotel bundle at a 12-15% discount increase
   booking conversion enough to offset the per-booking margin loss?"

Test Design:
  - Control (A): Hotel-only listing at full price
  - Treatment (B): Flight+Hotel bundle at 12-15% discount
  - Primary metric: Conversion rate (booked / shown)
  - Secondary metrics: Revenue per visitor, AOV, segment lift

Statistical Methods:
  - Two-proportion z-test (primary)
  - Chi-square test of independence
  - Bootstrap confidence intervals for conversion difference
  - Cohen's h effect size
  - Power analysis
  - Sequential testing (optional monitoring)
"""

import pandas as pd
import numpy as np
from typing import Optional
from scipy import stats

try:
    from config.settings import (
        FARE_RANGES, AB_TEST_SAMPLE_SIZE, AB_TEST_SEED,
    )
except ImportError:
    FARE_RANGES = {
        "economy":  {"min": 400,  "max": 900,   "mean": 620,  "std": 130},
        "business": {"min": 2500, "max": 6000,  "mean": 3800, "std": 900},
        "first":    {"min": 8000, "max": 20000, "mean": 12000, "std": 3000},
    }
    AB_TEST_SAMPLE_SIZE = 10000
    AB_TEST_SEED = 42

def load_ab_test_data() -> pd.DataFrame:
    """Load A/B test data from seeds (replaces generate_ab_test_data)."""
    from config.settings import SEEDS_DIR
    path = SEEDS_DIR / "ab_test.parquet"
    if path.exists():
        return pd.read_parquet(path)
    csv_path = SEEDS_DIR / "ab_test.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"No A/B test seed data found. Run: python scripts/generate_seeds.py"
    )

# ═══════════════════════════════════════════════════════════════
# PRE-TEST VALIDATION (SRM CHECK)
# ═══════════════════════════════════════════════════════════════

def check_sample_ratio_mismatch(df: pd.DataFrame) -> dict:
    """
    Sample Ratio Mismatch (SRM) test — verifies randomization is clean.

    If p < 0.01, the split is suspicious (instrumentation bug, bot traffic, etc.)
    """
    counts = df["GROUP"].value_counts()
    n_control = counts.get("control", 0)
    n_treatment = counts.get("treatment", 0)
    n_total = n_control + n_treatment

    expected = n_total / 2
    chi2 = ((n_control - expected) ** 2 + (n_treatment - expected) ** 2) / expected
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    result = {
        "n_control": n_control,
        "n_treatment": n_treatment,
        "expected_each": expected,
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 4),
        "srm_detected": p_value < 0.01,
        "verdict": "⚠️ SRM DETECTED — check randomization" if p_value < 0.01
                   else "✅ No SRM — randomization looks clean",
    }

    print(f"SRM Check: χ²={result['chi2_statistic']}, p={result['p_value']}")
    print(f"  {result['verdict']}")
    return result


def check_covariate_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify that covariates are balanced between control and treatment.
    Uses chi-square for categorical and t-test for numerical features.
    """
    results = []

    # Categorical balance
    for col in ["FARE_CLASS", "TRAVELER_TYPE", "DEVICE"]:
        ct = pd.crosstab(df["GROUP"], df[col])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        results.append({
            "COVARIATE": col,
            "TEST": "chi-square",
            "STATISTIC": round(chi2, 4),
            "P_VALUE": round(p, 4),
            "BALANCED": p > 0.05,
        })

    # Numerical balance
    for col in ["HOTEL_PRICE", "LEAD_TIME_DAYS"]:
        control_vals = df[df["GROUP"] == "control"][col]
        treatment_vals = df[df["GROUP"] == "treatment"][col]
        t_stat, p = stats.ttest_ind(control_vals, treatment_vals)
        results.append({
            "COVARIATE": col,
            "TEST": "t-test",
            "STATISTIC": round(t_stat, 4),
            "P_VALUE": round(p, 4),
            "BALANCED": p > 0.05,
        })

    balance_df = pd.DataFrame(results)
    n_imbalanced = (~balance_df["BALANCED"]).sum()
    print(f"Covariate balance: {len(balance_df) - n_imbalanced}/{len(balance_df)} balanced")
    return balance_df


# ═══════════════════════════════════════════════════════════════
# PRIMARY ANALYSIS: CONVERSION RATE
# ═══════════════════════════════════════════════════════════════

def run_conversion_test(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Primary A/B test: two-proportion z-test on conversion rates.

    Returns dict with all key metrics, test statistics, and verdict.
    """
    control = df[df["GROUP"] == "control"]
    treatment = df[df["GROUP"] == "treatment"]

    n_c, n_t = len(control), len(treatment)
    x_c = control["CONVERTED"].sum()
    x_t = treatment["CONVERTED"].sum()
    p_c = x_c / n_c
    p_t = x_t / n_t

    # Pooled proportion
    p_pool = (x_c + x_t) / (n_c + n_t)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))

    z_stat = (p_t - p_c) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # two-tailed

    # Confidence interval for the difference
    se_diff = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    ci_lower = diff - z_crit * se_diff
    ci_upper = diff + z_crit * se_diff

    # Relative lift
    relative_lift = (p_t - p_c) / p_c if p_c > 0 else 0

    # Effect size: Cohen's h
    cohens_h = 2 * (np.arcsin(np.sqrt(p_t)) - np.arcsin(np.sqrt(p_c)))

    result = {
        "control_n": n_c,
        "treatment_n": n_t,
        "control_conversions": x_c,
        "treatment_conversions": x_t,
        "control_cvr": round(p_c, 5),
        "treatment_cvr": round(p_t, 5),
        "absolute_diff": round(diff, 5),
        "relative_lift_pct": round(relative_lift * 100, 2),
        "ci_lower": round(ci_lower, 5),
        "ci_upper": round(ci_upper, 5),
        "z_statistic": round(z_stat, 4),
        "p_value": round(p_value, 6),
        "alpha": alpha,
        "significant": p_value < alpha,
        "cohens_h": round(cohens_h, 4),
        "effect_size_label": (
            "negligible" if abs(cohens_h) < 0.2 else
            "small" if abs(cohens_h) < 0.5 else
            "medium" if abs(cohens_h) < 0.8 else "large"
        ),
        "verdict": (
            f"🟢 SIGNIFICANT (p={p_value:.4f}): Bundle increases CVR by "
            f"{relative_lift:.1%} (absolute +{diff:.3%})"
            if p_value < alpha else
            f"🔴 NOT SIGNIFICANT (p={p_value:.4f}): No detectable difference"
        ),
    }

    print(f"\n{'='*60}")
    print(f"  PRIMARY A/B TEST RESULT")
    print(f"{'='*60}")
    print(f"  Control CVR:   {p_c:.3%}  ({x_c}/{n_c})")
    print(f"  Treatment CVR: {p_t:.3%}  ({x_t}/{n_t})")
    print(f"  Lift: {relative_lift:+.1%} absolute, {relative_lift*100:+.1f}% relative")
    print(f"  95% CI: [{ci_lower:+.4%}, {ci_upper:+.4%}]")
    print(f"  z={z_stat:.3f}, p={p_value:.5f}")
    print(f"  Cohen's h: {cohens_h:.4f} ({result['effect_size_label']})")
    print(f"\n  {result['verdict']}")

    return result


def run_chi_square_test(df: pd.DataFrame) -> dict:
    """
    Chi-square test of independence: GROUP × CONVERTED.
    Complements the z-test (should give identical p-value for 2×2).
    """
    ct = pd.crosstab(df["GROUP"], df["CONVERTED"])
    chi2, p_value, dof, expected = stats.chi2_contingency(ct)

    result = {
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 6),
        "dof": dof,
        "expected_frequencies": expected.round(2).tolist(),
        "significant": p_value < 0.05,
    }

    print(f"Chi-square: χ²={chi2:.4f}, p={p_value:.5f}, dof={dof}")
    return result


# ═══════════════════════════════════════════════════════════════
# BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════

def bootstrap_conversion_diff(
    df: pd.DataFrame,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = None,
) -> dict:
    """
    Non-parametric bootstrap CI for the conversion rate difference.
    More robust than normal approximation for small effect sizes.
    """
    seed = seed or AB_TEST_SEED
    rng = np.random.RandomState(seed)

    control = df[df["GROUP"] == "control"]["CONVERTED"].values
    treatment = df[df["GROUP"] == "treatment"]["CONVERTED"].values

    diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        c_sample = rng.choice(control, size=len(control), replace=True)
        t_sample = rng.choice(treatment, size=len(treatment), replace=True)
        diffs[i] = t_sample.mean() - c_sample.mean()

    alpha = 1 - confidence
    ci_lower = np.percentile(diffs, alpha / 2 * 100)
    ci_upper = np.percentile(diffs, (1 - alpha / 2) * 100)

    result = {
        "mean_diff": round(diffs.mean(), 5),
        "median_diff": round(np.median(diffs), 5),
        "std_diff": round(diffs.std(), 5),
        "ci_lower": round(ci_lower, 5),
        "ci_upper": round(ci_upper, 5),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "pct_positive": round((diffs > 0).mean() * 100, 2),
        "bootstrap_diffs": diffs,  # for plotting
    }

    print(f"Bootstrap ({n_bootstrap:,} iterations):")
    print(f"  Mean diff: {diffs.mean():+.4%}")
    print(f"  {confidence:.0%} CI: [{ci_lower:+.4%}, {ci_upper:+.4%}]")
    print(f"  P(treatment > control): {result['pct_positive']:.1f}%")

    return result


# ═══════════════════════════════════════════════════════════════
# REVENUE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_revenue(df: pd.DataFrame) -> dict:
    """
    Compare revenue metrics between groups.

    Key insight: Bundle may have higher CVR but lower per-booking revenue
    due to discount. Net effect on revenue-per-visitor is what matters.
    """
    control = df[df["GROUP"] == "control"]
    treatment = df[df["GROUP"] == "treatment"]

    # Revenue per visitor (RPV) — the metric that matters
    rpv_control = control["REVENUE"].sum() / len(control)
    rpv_treatment = treatment["REVENUE"].sum() / len(treatment)
    rpv_lift = (rpv_treatment - rpv_control) / rpv_control if rpv_control > 0 else 0

    # Average order value (only among converters)
    aov_control = control[control["CONVERTED"] == 1]["REVENUE"].mean()
    aov_treatment = treatment[treatment["CONVERTED"] == 1]["REVENUE"].mean()

    # Total revenue
    total_rev_control = control["REVENUE"].sum()
    total_rev_treatment = treatment["REVENUE"].sum()

    # Revenue by fare class
    rev_by_fare = (
        df.groupby(["GROUP", "FARE_CLASS"])
        .agg(
            TOTAL_REVENUE=("REVENUE", "sum"),
            CONVERSIONS=("CONVERTED", "sum"),
            VISITORS=("VISITOR_ID", "count"),
            AVG_REVENUE=("REVENUE", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        )
        .reset_index()
    )
    rev_by_fare["RPV"] = (rev_by_fare["TOTAL_REVENUE"] / rev_by_fare["VISITORS"]).round(2)
    rev_by_fare["CVR"] = (rev_by_fare["CONVERSIONS"] / rev_by_fare["VISITORS"]).round(4)

    result = {
        "rpv_control": round(rpv_control, 2),
        "rpv_treatment": round(rpv_treatment, 2),
        "rpv_lift_pct": round(rpv_lift * 100, 2),
        "aov_control": round(aov_control, 2),
        "aov_treatment": round(aov_treatment, 2),
        "aov_change_pct": round((aov_treatment - aov_control) / aov_control * 100, 2)
            if aov_control > 0 else 0,
        "total_revenue_control": round(total_rev_control, 2),
        "total_revenue_treatment": round(total_rev_treatment, 2),
        "revenue_by_fare": rev_by_fare,
    }

    print(f"\n{'='*60}")
    print(f"  REVENUE ANALYSIS")
    print(f"{'='*60}")
    print(f"  RPV Control:   ${rpv_control:,.2f}")
    print(f"  RPV Treatment: ${rpv_treatment:,.2f}  ({rpv_lift:+.1%})")
    print(f"  AOV Control:   ${aov_control:,.2f}")
    print(f"  AOV Treatment: ${aov_treatment:,.2f}  ({result['aov_change_pct']:+.1f}%)")
    print(f"\n  💡 Bundle discount lowers AOV but higher CVR {'increases' if rpv_lift > 0 else 'does NOT offset'} RPV")

    return result


# ═══════════════════════════════════════════════════════════════
# SEGMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Break down A/B test results by key segments.
    Identifies which traveler groups benefit most from bundling.
    """
    segments = []

    for segment_col in ["FARE_CLASS", "TRAVELER_TYPE", "DEVICE"]:
        for segment_val in df[segment_col].unique():
            seg_df = df[df[segment_col] == segment_val]
            control = seg_df[seg_df["GROUP"] == "control"]
            treatment = seg_df[seg_df["GROUP"] == "treatment"]

            if len(control) < 30 or len(treatment) < 30:
                continue

            cvr_c = control["CONVERTED"].mean()
            cvr_t = treatment["CONVERTED"].mean()
            lift = (cvr_t - cvr_c) / cvr_c if cvr_c > 0 else 0

            rpv_c = control["REVENUE"].sum() / len(control)
            rpv_t = treatment["REVENUE"].sum() / len(treatment)

            # Quick z-test per segment
            p_pool = seg_df["CONVERTED"].mean()
            se = np.sqrt(p_pool * (1 - p_pool) * (1/len(control) + 1/len(treatment)))
            z = (cvr_t - cvr_c) / se if se > 0 else 0
            p_val = 2 * (1 - stats.norm.cdf(abs(z)))

            segments.append({
                "SEGMENT_TYPE": segment_col,
                "SEGMENT_VALUE": segment_val,
                "N_CONTROL": len(control),
                "N_TREATMENT": len(treatment),
                "CVR_CONTROL": round(cvr_c, 4),
                "CVR_TREATMENT": round(cvr_t, 4),
                "RELATIVE_LIFT_PCT": round(lift * 100, 1),
                "RPV_CONTROL": round(rpv_c, 2),
                "RPV_TREATMENT": round(rpv_t, 2),
                "P_VALUE": round(p_val, 4),
                "SIGNIFICANT": p_val < 0.05,
            })

    seg_df = pd.DataFrame(segments).sort_values("RELATIVE_LIFT_PCT", ascending=False)

    print(f"\nSegment Analysis: {len(seg_df)} segments tested")
    sig_count = seg_df["SIGNIFICANT"].sum()
    print(f"  Significant segments: {sig_count}/{len(seg_df)}")

    return seg_df


# ═══════════════════════════════════════════════════════════════
# SEQUENTIAL TESTING (MONITORING SIMULATION)
# ═══════════════════════════════════════════════════════════════

def simulate_sequential_test(
    df: pd.DataFrame,
    check_points: int = 20,
    alpha_spending: str = "obrien_fleming",
) -> pd.DataFrame:
    """
    Simulate sequential monitoring of the A/B test.

    Shows what the test result would have been at each checkpoint.
    Uses O'Brien-Fleming-like alpha spending to control false positives.
    """
    n = len(df)
    step_size = n // check_points
    alpha_total = 0.05

    results = []
    for i in range(1, check_points + 1):
        idx = min(i * step_size, n)
        partial = df.iloc[:idx]

        control = partial[partial["GROUP"] == "control"]
        treatment = partial[partial["GROUP"] == "treatment"]

        if len(control) < 10 or len(treatment) < 10:
            continue

        cvr_c = control["CONVERTED"].mean()
        cvr_t = treatment["CONVERTED"].mean()
        diff = cvr_t - cvr_c

        p_pool = partial["CONVERTED"].mean()
        se = np.sqrt(p_pool * (1 - p_pool) * (1/len(control) + 1/len(treatment)))
        z = diff / se if se > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))

        # O'Brien-Fleming spending: very conservative early, relaxed late
        info_fraction = i / check_points
        if alpha_spending == "obrien_fleming":
            z_boundary = stats.norm.ppf(1 - alpha_total / (2 * np.sqrt(info_fraction)))
            spent_alpha = 2 * (1 - stats.norm.cdf(z_boundary))
        else:
            spent_alpha = alpha_total * info_fraction  # Pocock

        results.append({
            "CHECKPOINT": i,
            "N_OBSERVED": idx,
            "PCT_COMPLETE": round(idx / n * 100, 1),
            "CVR_CONTROL": round(cvr_c, 4),
            "CVR_TREATMENT": round(cvr_t, 4),
            "DIFF": round(diff, 5),
            "Z_STAT": round(z, 3),
            "P_VALUE": round(p_val, 5),
            "ALPHA_BOUNDARY": round(spent_alpha, 5),
            "WOULD_STOP": p_val < spent_alpha,
        })

    seq_df = pd.DataFrame(results)
    first_stop = seq_df[seq_df["WOULD_STOP"]]
    if not first_stop.empty:
        stop_at = first_stop.iloc[0]
        print(f"Sequential: Would stop at checkpoint {stop_at['CHECKPOINT']} "
              f"({stop_at['PCT_COMPLETE']}% data, p={stop_at['P_VALUE']})")
    else:
        print("Sequential: Would NOT stop early at any checkpoint")

    return seq_df


# ═════════════��═════════════════════════════════════════════════
# POWER ANALYSIS
# ═══════════════════════════════════════════════════════════════

def compute_required_sample_size(
    baseline_cvr: float = 0.032,
    min_detectable_lift: float = 0.20,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Compute minimum sample size per group for a two-proportion z-test.
    """
    p1 = baseline_cvr
    p2 = baseline_cvr * (1 + min_detectable_lift)

    # Cohen's h
    h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    n_per_group = int(np.ceil(((z_alpha + z_beta) / h) ** 2))
    n_total = n_per_group * 2

    result = {
        "baseline_cvr": p1,
        "target_cvr": round(p2, 5),
        "min_detectable_lift_pct": min_detectable_lift * 100,
        "cohens_h": round(h, 4),
        "alpha": alpha,
        "power": power,
        "n_per_group": n_per_group,
        "n_total": n_total,
        "current_n": AB_TEST_SAMPLE_SIZE,
        "adequately_powered": AB_TEST_SAMPLE_SIZE >= n_total,
    }

    print(f"\nPower Analysis:")
    print(f"  Baseline CVR: {p1:.2%} → Target: {p2:.2%} (MDE: {min_detectable_lift:.0%} lift)")
    print(f"  Required: {n_per_group:,}/group = {n_total:,} total")
    print(f"  Current:  {AB_TEST_SAMPLE_SIZE:,}")
    print(f"  {'✅ Adequately powered' if result['adequately_powered'] else '⚠️ UNDERPOWERED'}")

    return result


# ═══════════════════════════════════════════════════════════════
# INCREMENTAL REVENUE PROJECTION
# ═══════════════════════════════════════════════════════════════

def project_incremental_revenue(
    test_result: dict,
    revenue_result: dict,
    monthly_visitors: int = 500_000,
) -> pd.DataFrame:
    """
    Project annualized incremental revenue from rolling out the bundle.
    """
    if not test_result["significant"]:
        print("⚠️ Test not significant — projections are speculative")

    rpv_lift = revenue_result["rpv_treatment"] - revenue_result["rpv_control"]

    scenarios = []
    for scenario, traffic_mult in [("Conservative", 0.7), ("Base", 1.0), ("Optimistic", 1.3)]:
        monthly_inc = rpv_lift * monthly_visitors * traffic_mult
        annual_inc = monthly_inc * 12

        scenarios.append({
            "SCENARIO": scenario,
            "MONTHLY_VISITORS": int(monthly_visitors * traffic_mult),
            "RPV_LIFT": round(rpv_lift, 2),
            "MONTHLY_INCREMENTAL_REV": round(monthly_inc, 2),
            "ANNUAL_INCREMENTAL_REV": round(annual_inc, 2),
        })

    proj_df = pd.DataFrame(scenarios)

    print(f"\n{'='*60}")
    print(f"  REVENUE PROJECTION (if rolled out)")
    print(f"{'='*60}")
    for _, row in proj_df.iterrows():
        print(f"  {row['SCENARIO']:>12}: ${row['ANNUAL_INCREMENTAL_REV']:>12,.0f}/year "
              f"(${row['MONTHLY_INCREMENTAL_REV']:>10,.0f}/month)")

    return proj_df


# ════════════════════════════════════════��══════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_full_ab_analysis(df: pd.DataFrame = None) -> dict:
    """Run the complete A/B test analysis pipeline."""
    if df is None:
        df = generate_ab_test_data()

    print("\n" + "=" * 60)
    print("  M03: A/B TEST — BUNDLE vs HOTEL-ONLY PRICING")
    print("=" * 60)

    # 1. Pre-test checks
    srm = check_sample_ratio_mismatch(df)
    balance = check_covariate_balance(df)

    # 2. Primary test
    test_result = run_conversion_test(df)
    chi2_result = run_chi_square_test(df)

    # 3. Bootstrap
    bootstrap = bootstrap_conversion_diff(df)

    # 4. Revenue
    revenue = analyze_revenue(df)

    # 5. Segments
    segments = analyze_segments(df)

    # 6. Sequential
    sequential = simulate_sequential_test(df)

    # 7. Power
    power = compute_required_sample_size()

    # 8. Projection
    projection = project_incremental_revenue(test_result, revenue)

    return {
        "data": df,
        "srm_check": srm,
        "covariate_balance": balance,
        "test_result": test_result,
        "chi2_result": chi2_result,
        "bootstrap": bootstrap,
        "revenue": revenue,
        "segments": segments,
        "sequential": sequential,
        "power": power,
        "projection": projection,
    }


# ═══════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_full_ab_analysis()
    print("\n✅ M03 A/B Test pipeline complete")