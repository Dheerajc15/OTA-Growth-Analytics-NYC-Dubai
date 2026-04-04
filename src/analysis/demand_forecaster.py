"""
Demand Forecasting Engine (Module 01)
=======================================
Data Sources: Google Trends + Aviation Edge

Answers: "When should a platform push Dubai inventory to NYC users —
and how far in advance?"

Models:
  1. Prophet — Ramadan holidays + Google Trends regressors + capacity supply
  2. SARIMA — classical baseline with monthly seasonality
"""

import pandas as pd
import numpy as np
from typing import Optional
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from src.preprocessing.aviation import prepare_forecast_data


# ═══════════════════════════════════════════════════════════════
# RAMADAN + HOLIDAY CALENDAR
# ═══════════════════════════════════════════════════════════════

def get_dubai_holidays() -> pd.DataFrame:
    """Ramadan, Eid, and Dubai high-season holidays."""
    ramadan = pd.DataFrame({
        "holiday": "ramadan",
        "ds": pd.to_datetime([
            "2019-05-06", "2020-04-24", "2021-04-13", "2022-04-02",
            "2023-03-23", "2024-03-12", "2025-03-01", "2026-02-18",
        ]),
        "lower_window": -7,
        "upper_window": 35,
    })

    eid = pd.DataFrame({
        "holiday": "eid_al_fitr",
        "ds": pd.to_datetime([
            "2019-06-04", "2020-05-24", "2021-05-13", "2022-05-02",
            "2023-04-21", "2024-04-10", "2025-03-31", "2026-03-20",
        ]),
        "lower_window": -3,
        "upper_window": 7,
    })

    nye = pd.DataFrame({
        "holiday": "nye_dubai_season",
        "ds": pd.to_datetime([f"{y}-12-20" for y in range(2019, 2027)]),
        "lower_window": -5,
        "upper_window": 15,
    })

    return pd.concat([ramadan, eid, nye], ignore_index=True)


# ═══════════════════════════════════════════════════════════════
# PROPHET MODEL
# ═══════════════════════════════════════════════════════════════

def train_prophet_model(
    df: pd.DataFrame,
    regressor_cols: Optional[list[str]] = None,
    test_months: int = 6,
    forecast_months: int = 12,
) -> dict:
    """
    Train Prophet with holidays + supply/demand regressors.

    Returns dict: model, forecast, metrics, train, test
    """
    from prophet import Prophet

    if regressor_cols is None:
        regressor_cols = [c for c in df.columns
                         if c.startswith("trend_") or c in ["supply_flights", "load_factor"]]

    train = df.iloc[:-test_months].copy()
    test = df.iloc[-test_months:].copy()

    print(f"Prophet — Train: {len(train)} months, Test: {len(test)} months")
    print(f"  Regressors: {regressor_cols}")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
        interval_width=0.95,
    )
    model.holidays = get_dubai_holidays()

    for col in regressor_cols:
        model.add_regressor(col, standardize=True)

    model.fit(train)

    # Future dataframe
    future = model.make_future_dataframe(periods=test_months + forecast_months, freq="MS")

    for col in regressor_cols:
        future = future.merge(df[["ds", col]], on="ds", how="left")
        future[col] = future[col].ffill().bfill()

    forecast = model.predict(future)

    # Evaluate
    test_mask = forecast["ds"].isin(test["ds"])
    pred = forecast[test_mask].set_index("ds")["yhat"]
    actual = test.set_index("ds")["y"]
    common = actual.index.intersection(pred.index)

    if len(common) > 0:
        mae = mean_absolute_error(actual[common], pred[common])
        mape = mean_absolute_percentage_error(actual[common], pred[common])
    else:
        mae, mape = np.nan, np.nan

    print(f"\n  📊 MAE: {mae:,.0f}  |  MAPE: {mape:.1%}")

    return {
        "model": model,
        "forecast": forecast,
        "metrics": {"mae": mae, "mape": mape, "mape_pct": mape * 100 if not np.isnan(mape) else np.nan},
        "train": train,
        "test": test,
    }


# ═══════════════════════════════════════════════════════════════
# SARIMA BASELINE
# ═══════════════════════════════════════════════════════════════

def train_sarima_model(
    df: pd.DataFrame,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 12),
    test_months: int = 6,
) -> dict:
    """SARIMA baseline with monthly seasonality (s=12)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    train = df.iloc[:-test_months].copy()
    test = df.iloc[-test_months:].copy()

    print(f"SARIMA{order}×{seasonal_order} — Train: {len(train)}, Test: {len(test)}")

    model = SARIMAX(train["y"], order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=500)

    forecast = fitted.forecast(steps=test_months)
    mae = mean_absolute_error(test["y"], forecast)
    mape = mean_absolute_percentage_error(test["y"], forecast)

    print(f"  📊 MAE: {mae:,.0f}  |  MAPE: {mape:.1%}  |  AIC: {fitted.aic:.0f}")

    return {
        "model": fitted,
        "forecast": forecast,
        "metrics": {"mae": mae, "mape": mape, "mape_pct": mape * 100},
        "aic": fitted.aic,
        "train": train,
        "test": test,
    }


# ═══════════════════════════════════════════════════════════════
# SEARCH → DEMAND LAG ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_search_demand_lag(
    merged_df: pd.DataFrame,
    search_col: str,
    demand_col: str = "y",
    max_lag: int = 6,
) -> pd.DataFrame:
    """Test if Google Trends leads passenger demand by 0–N months."""
    from scipy.stats import pearsonr

    results = []
    for lag in range(0, max_lag + 1):
        shifted = merged_df[search_col].shift(lag)
        valid = pd.DataFrame({"search": shifted, "demand": merged_df[demand_col]}).dropna()

        if len(valid) < 5:
            continue

        corr, p = pearsonr(valid["search"], valid["demand"])
        results.append({
            "lag_months": lag,
            "description": f"Search leads by {lag} month(s)" if lag > 0 else "Same month",
            "correlation": round(corr, 4),
            "p_value": round(p, 4),
            "significant": p < 0.05,
            "n": len(valid),
        })

    lag_df = pd.DataFrame(results)
    if not lag_df.empty:
        best = lag_df.loc[lag_df["correlation"].abs().idxmax()]
        print(f"🔍 Best lag: {best['lag_months']} month(s) (r={best['correlation']}, p={best['p_value']})")
    return lag_df


# ═══════════════════════════════════════════════════════════════
# PUSH TIMING RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════

def generate_push_timing(forecast: pd.DataFrame, lead_months: int = 3) -> pd.DataFrame:
    """When should an OTA start pushing Dubai inventory?"""
    future = forecast[forecast["ds"] > pd.Timestamp.now()].copy()
    if future.empty:
        return pd.DataFrame()

    threshold = future["yhat"].quantile(0.75)
    peaks = future[future["yhat"] >= threshold].copy()
    peaks["PUSH_START"] = peaks["ds"] - pd.DateOffset(months=lead_months)
    peaks["PUSH_END"] = peaks["ds"] - pd.DateOffset(months=1)

    def season(dt):
        m = dt.month
        if m in [12, 1, 2]: return "Winter Peak (NYE/Dubai Season)"
        elif m in [3, 4, 5]: return "Spring (Post-Ramadan)"
        elif m in [6, 7, 8]: return "Summer Low"
        else: return "Fall Kickoff"

    peaks["SEASON"] = peaks["ds"].apply(season)
    peaks["RECOMMENDATION"] = peaks.apply(
        lambda r: f"🚀 Push inventory {r['PUSH_START'].strftime('%b %Y')}–"
                  f"{r['PUSH_END'].strftime('%b %Y')} → peak {r['ds'].strftime('%b %Y')} "
                  f"({r['yhat']:,.0f} est. passengers). {r['SEASON']}.",
        axis=1,
    )

    return peaks[["ds", "yhat", "yhat_lower", "yhat_upper",
                   "PUSH_START", "PUSH_END", "SEASON", "RECOMMENDATION"]].rename(
        columns={"ds": "PEAK_MONTH", "yhat": "FORECAST_PAX"}
    ).reset_index(drop=True)


def compare_models(prophet_res: dict, sarima_res: dict) -> pd.DataFrame:
    comparison = pd.DataFrame([
        {"Model": "Prophet (holidays + regressors)", "MAE": prophet_res["metrics"]["mae"],
         "MAPE%": prophet_res["metrics"]["mape_pct"]},
        {"Model": "SARIMA (baseline)", "MAE": sarima_res["metrics"]["mae"],
         "MAPE%": sarima_res["metrics"]["mape_pct"], "AIC": sarima_res["aic"]},
    ])
    winner = "Prophet" if prophet_res["metrics"]["mape"] < sarima_res["metrics"]["mape"] else "SARIMA"
    print(f"\n🏆 Winner: {winner}")
    print(comparison.to_string(index=False))
    return comparison


# ═══════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.data_collection.aviation_edge import generate_synthetic_monthly_capacity
    from src.data_collection.google_trends import generate_synthetic_trends

    print("Demand Forecaster — Full Pipeline Test")
    print("=" * 50)

    capacity = generate_synthetic_monthly_capacity()
    trends = generate_synthetic_trends()

    prophet_df = prepare_forecast_data(capacity, trends)

    print("\n--- Prophet ---")
    p_res = train_prophet_model(prophet_df, test_months=6, forecast_months=12)

    print("\n--- SARIMA ---")
    s_res = train_sarima_model(prophet_df, test_months=6)

    compare_models(p_res, s_res)

    print("\n--- Push Timing ---")
    recs = generate_push_timing(p_res["forecast"])
    if not recs.empty:
        print(recs[["PEAK_MONTH", "FORECAST_PAX", "PUSH_START", "SEASON"]].to_string(index=False))