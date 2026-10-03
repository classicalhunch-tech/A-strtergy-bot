"""
market_data/mtf_features.py

Multi-timeframe feature helpers built on top of the existing
market_data/features.py imbalance utilities and the MTF context attached by
strategy/mtf_structure.py and dashboard/mtf_context.py.

This file is intentionally small, causal, and easy to drop into an existing
strategy pipeline. It does not alter execution logic; it simply exposes a
clean way to read MTF context and attach features to signals.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def get_mtf_trend(df: pd.DataFrame, prefix: str, timestamp: Optional[pd.Timestamp] = None) -> Optional[str]:
    """Return the current higher-timeframe trend for a row or timestamp.

    Expected columns: {prefix}trend, e.g. "macro_trend" or "internal_trend".
    """
    if prefix + "trend" not in df.columns:
        return None

    if timestamp is None:
        value = df[prefix + "trend"].iloc[-1]
    else:
        if timestamp not in df.index:
            # nearest timestamp fallback for causal lookup
            idx = df.index.get_indexer([timestamp], method="nearest")[0]
            value = df[prefix + "trend"].iloc[idx]
        else:
            value = df.loc[timestamp, prefix + "trend"]

    if pd.isna(value):
        return None
    return str(value).upper()


def compute_mtf_alignment(
    df: pd.DataFrame,
    prefix: str,
    signal_direction: str,
    timestamp: Optional[pd.Timestamp] = None,
) -> bool:
    """Return True when the signal direction aligns with the HTF trend.

    Example signal_direction values: "LONG", "SHORT".
    """
    trend = get_mtf_trend(df, prefix, timestamp=timestamp)
    if trend is None:
        return False

    if signal_direction.upper() == "LONG" and trend == "BULLISH":
        return True
    if signal_direction.upper() == "SHORT" and trend == "BEARISH":
        return True
    return False


def get_recent_imbalance(
    df: pd.DataFrame,
    imbalance_col: str,
    timestamp: Optional[pd.Timestamp] = None,
    lookback: int = 10,
) -> float:
    """Return a trailing average imbalance from recent 5m rows.

    This is a simple causal roll-up for the current timestamp. It uses only
    values at or before the given time.
    """
    if imbalance_col not in df.columns:
        return np.nan

    if timestamp is None:
        series = df[imbalance_col].dropna().tail(lookback)
    else:
        if timestamp not in df.index:
            idx = df.index.get_indexer([timestamp], method="nearest")[0]
        else:
            idx = df.index.get_loc(timestamp)
        series = df.iloc[: idx + 1][imbalance_col].dropna().tail(lookback)

    if series.empty:
        return np.nan
    return float(series.mean())


def build_mtf_feature_row(
    df: pd.DataFrame,
    timestamp: Optional[pd.Timestamp] = None,
    signal_direction: Optional[str] = None,
    prefix: str = "macro_",
    imbalance_col: str = "imbalance_n500",
) -> Dict[str, Any]:
    """Extract a feature dict for one point in time.

    The result is small and easy to attach to a strategy decision.
    """
    if timestamp is None:
        timestamp = df.index[-1]

    trend = get_mtf_trend(df, prefix, timestamp=timestamp)
    aligned = False
    if signal_direction is not None:
        aligned = compute_mtf_alignment(df, prefix, signal_direction, timestamp=timestamp)

    return {
        "timestamp": timestamp,
        "trend": trend,
        "alignment": aligned,
        "imbalance": get_recent_imbalance(df, imbalance_col, timestamp=timestamp, lookback=10),
    }


def add_mtf_feature_columns(
    df: pd.DataFrame,
    prefix: str = "macro_",
    imbalance_col: str = "imbalance_n500",
) -> pd.DataFrame:
    """Attach a few MTF feature columns to a dataframe in-place via a copy.

    Adds:
      - {prefix}trend
      - {prefix}alignment
      - {prefix}imbalance
    """
    out = df.copy()

    trend_values = []
    aligned_values = []
    imbalance_values = []

    for ts in out.index:
        trend = get_mtf_trend(out, prefix, timestamp=ts)
        trend_values.append(trend)

        if "signal_direction" in out.columns:
            signal_direction = out.loc[ts, "signal_direction"]
            aligned_values.append(compute_mtf_alignment(out, prefix, str(signal_direction), timestamp=ts))
        else:
            aligned_values.append(False)

        imbalance_values.append(get_recent_imbalance(out, imbalance_col, timestamp=ts, lookback=10))

    out[prefix + "trend"] = trend_values
    out[prefix + "alignment"] = aligned_values
    out[prefix + "imbalance"] = imbalance_values
    return out
