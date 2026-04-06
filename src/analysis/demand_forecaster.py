
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


def _validate_ts(df: pd.DataFrame, date_col: str, y_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")

    out = out.dropna(subset=[date_col]).sort_values(date_col)
    out = out[[date_col, y_col]].drop_duplicates(subset=[date_col]).reset_index(drop=True)

    if out.empty:
        raise ValueError("Time series is empty after parsing.")
    return out


def to_monthly_series(
    df: pd.DataFrame,
    date_col: str = "DATE",
    y_col: str = "VALUE",
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Convert raw timestamp data into monthly series at month-start frequency (MS).
    """
    out = _validate_ts(df, date_col, y_col).set_index(date_col)

    if agg == "sum":
        monthly = out[y_col].resample("MS").sum()
    else:
        monthly = out[y_col].resample("MS").mean()

    monthly = monthly.interpolate(limit_direction="both")
    monthly = monthly.reset_index().rename(columns={y_col: "Y"})
    return monthly


def train_test_split_time(
    monthly_df: pd.DataFrame,
    test_periods: int = 12,
    date_col: str = "DATE",
    y_col: str = "Y",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(monthly_df) <= test_periods + 6:
        raise ValueError("Not enough history for stable split.")

    df = monthly_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)

    train = df.iloc[:-test_periods].copy()
    test = df.iloc[-test_periods:].copy()
    return train, test


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-9, 1.0, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = _safe_mape(y_true, y_pred)

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}


def forecast_naive(train: pd.DataFrame, test: pd.DataFrame, y_col: str = "Y") -> np.ndarray:
    last = float(train[y_col].iloc[-1])
    return np.full(len(test), last)


def forecast_seasonal_naive(
    train: pd.DataFrame,
    test: pd.DataFrame,
    season_length: int = 12,
    y_col: str = "Y",
) -> np.ndarray:
    if len(train) < season_length:
        return forecast_naive(train, test, y_col=y_col)

    hist = train[y_col].values
    preds = []
    for i in range(len(test)):
        idx = len(hist) - season_length + (i % season_length)
        preds.append(float(hist[idx]))
    return np.array(preds)


def forecast_linear_trend(train: pd.DataFrame, test: pd.DataFrame, y_col: str = "Y") -> np.ndarray:
    y = train[y_col].values.astype(float)
    x = np.arange(len(y), dtype=float)

    # Simple OLS via polyfit (degree 1)
    slope, intercept = np.polyfit(x, y, 1)
    x_future = np.arange(len(y), len(y) + len(test), dtype=float)
    pred = intercept + slope * x_future
    return pred.astype(float)


def run_forecast_benchmarks(
    monthly_df: pd.DataFrame,
    date_col: str = "DATE",
    y_col: str = "Y",
    test_periods: int = 12,
) -> dict[str, ForecastResult]:
    train, test = train_test_split_time(
        monthly_df.rename(columns={date_col: "DATE", y_col: "Y"}),
        test_periods=test_periods,
        date_col="DATE",
        y_col="Y",
    )

    models = {
        "naive": forecast_naive(train, test, y_col="Y"),
        "seasonal_naive": forecast_seasonal_naive(train, test, season_length=12, y_col="Y"),
        "linear_trend": forecast_linear_trend(train, test, y_col="Y"),
    }

    out = {}
    for name, pred in models.items():
        metrics = evaluate_forecast(test["Y"].values, pred)
        pred_df = test[["DATE"]].copy()
        pred_df["ACTUAL"] = test["Y"].values
        pred_df["PRED"] = pred
        pred_df["RESIDUAL"] = pred_df["ACTUAL"] - pred_df["PRED"]

        out[name] = ForecastResult(
            model_name=name,
            train_end=pd.Timestamp(train["DATE"].max()),
            test_start=pd.Timestamp(test["DATE"].min()),
            test_end=pd.Timestamp(test["DATE"].max()),
            metrics=metrics,
            predictions=pred_df,
        )
    return out


def best_model_summary(results: dict[str, ForecastResult], metric: str = "RMSE") -> pd.DataFrame:
    rows = []
    for k, v in results.items():
        row = {"MODEL": k}
        row.update(v.metrics)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(metric, ascending=True).reset_index(drop=True)
    return df