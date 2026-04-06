from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    model_name: str
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: dict
    predictions: pd.DataFrame


def train_test_split_time(df: pd.DataFrame, test_periods: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df.copy()
    x["ds"] = pd.to_datetime(x["ds"], errors="coerce")
    x["y"] = pd.to_numeric(x["y"], errors="coerce")
    x = x.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

    if len(x) <= test_periods + 6:
        raise ValueError("Not enough observations for stable split.")
    return x.iloc[:-test_periods], x.iloc[-test_periods:]


def _metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.where(np.abs(y_true) < 1e-9, 1.0, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}


def run_forecast_benchmarks(forecast_ready_df: pd.DataFrame, test_periods: int = 12) -> dict[str, ForecastResult]:
    train, test = train_test_split_time(forecast_ready_df, test_periods=test_periods)
    y_train = train["y"].values
    y_test = test["y"].values

    naive = np.full(len(test), y_train[-1])

    season_len = 12
    if len(y_train) >= season_len:
        seasonal = np.array([y_train[-season_len + (i % season_len)] for i in range(len(test))], dtype=float)
    else:
        seasonal = naive.copy()

    x = np.arange(len(y_train), dtype=float)
    m, b = np.polyfit(x, y_train, 1)
    trend = b + m * np.arange(len(y_train), len(y_train) + len(test), dtype=float)

    preds = {"naive": naive, "seasonal_naive": seasonal, "linear_trend": trend}
    out = {}

    for k, p in preds.items():
        pred_df = test[["ds", "y"]].copy()
        pred_df["pred"] = p
        pred_df["residual"] = pred_df["y"] - pred_df["pred"]
        out[k] = ForecastResult(
            model_name=k,
            train_end=train["ds"].max(),
            test_start=test["ds"].min(),
            test_end=test["ds"].max(),
            metrics=_metrics(y_test, p),
            predictions=pred_df,
        )

    return out