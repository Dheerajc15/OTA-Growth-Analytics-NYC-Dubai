from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.hotels import prepare_funnel_data as preprocess_hotels


def compare_markets(df_prepared: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market in df_prepared["MARKET"].dropna().unique():
        m = df_prepared[df_prepared["MARKET"] == market]
        b = m[m["IS_BOOKABLE"]]
        rows.append({
            "MARKET": market,
            "TOTAL_LISTINGS": len(m),
            "BOOKABLE_LISTINGS": len(b),
            "BOOKABILITY_RATE": round(len(b) / max(len(m), 1) * 100, 1),
            "AVG_PRICE_LEVEL": round(float(m["PRICE_LEVEL"].mean()), 2),
            "AVG_RATING": round(float(m["RATING"].mean()), 2),
            "AVG_VISIBILITY": round(float(m["VISIBILITY_SCORE"].mean()), 1),
        })
    return pd.DataFrame(rows)


def simulate_booking_funnel(df_prepared: pd.DataFrame, market: str, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    mdf = df_prepared[df_prepared["MARKET"] == market].copy()
    if mdf.empty:
        return pd.DataFrame()

    rates = {"search_to_view": 0.55, "view_to_compare": 0.35, "compare_to_intent": 0.50, "intent_to_book": 0.65} \
        if market == "Dubai" else \
        {"search_to_view": 0.40, "view_to_compare": 0.50, "compare_to_intent": 0.55, "intent_to_book": 0.70}

    mdf["P_VIEW"] = np.clip(rates["search_to_view"] * (mdf["VISIBILITY_SCORE"] / 50), 0.05, 0.95)
    mdf["P_COMPARE"] = np.clip(rates["view_to_compare"] * (mdf["RATING"].fillna(3.0) / 4.0), 0.05, 0.95)
    mdf["P_INTENT"] = np.clip(rates["compare_to_intent"] * (1 - mdf["PRICE_LEVEL"].fillna(2.0) / 6), 0.05, 0.90)
    mdf["P_BOOK"] = np.clip(rates["intent_to_book"] * (mdf["TOTAL_RATINGS"].clip(0, 5000) / 3000), 0.05, 0.90)

    total_visitors = 10000
    share = mdf["VISIBILITY_SCORE"] / max(float(mdf["VISIBILITY_SCORE"].sum()), 1.0)

    mdf["STAGE_1_SEARCH"] = (total_visitors * share).round().astype(int).clip(lower=1)
    mdf["STAGE_2_VIEW"] = (mdf["STAGE_1_SEARCH"] * mdf["P_VIEW"]).astype(int)
    mdf["STAGE_3_COMPARE"] = (mdf["STAGE_2_VIEW"] * mdf["P_COMPARE"]).astype(int)
    mdf["STAGE_4_INTENT"] = (mdf["STAGE_3_COMPARE"] * mdf["P_INTENT"]).astype(int)
    mdf["STAGE_5_BOOK"] = (mdf["STAGE_4_INTENT"] * mdf["P_BOOK"]).astype(int)
    return mdf