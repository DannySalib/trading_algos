from __future__ import annotations

import hashlib
import json

import pandas as pd

from .forecast import Forecast
from ...models.period import Period
from ...models.signal import Signal
from ...config import CACHE_PATH
from ...cache_functions import is_stale, cache_get, cache_put

AVG_ABS_MIN_PERIOD = 10
_LAG_DAYS = 1
_CACHE_PATH_FORECASTS = CACHE_PATH / "forecasts"
_CACHE_TTL = 60 * 60 * 24 * 30 * AVG_ABS_MIN_PERIOD

def lag(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return data.shift(_LAG_DAYS) # remove lookahead bias in ONE place

def construct_forecasts(signals: list[Signal]) -> list[Forecast]:
    try:
        return [Forecast(s) for s in signals]
    except (AttributeError, TypeError, ValueError) as e:
        raise RuntimeError("Failed to initialize forecasts") from e

def forecast_scalar_key(
    forecasts: list[Forecast],
    tickers: list[str],
    period: Period,
) -> str:
    """Cache key for the forecast scalar.

    Built from the forecast names, the (sorted) universe tickers and the requested
    period (t0/tf, not the padded one), so changing any of them gets a fresh scalar
    instead of reusing one that doesn't match the data. Hashed because a large
    universe would make an unwieldy shelve key.
    """
    payload = {
        "forecasts": sorted(f.name for f in forecasts),
        "tickers": sorted(tickers),
        "t0": period.t0.isoformat(),
        "tf": period.tf.isoformat(),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

def stale_forecast_scalar(key: str) -> bool:
    return is_stale(
        key=key,
        cache_path=str(_CACHE_PATH_FORECASTS),
        ttl=_CACHE_TTL,
    )

def apply_forecast_scalar(
    df: pd.DataFrame,
    key: str,
    stale_scalar: bool,
) -> pd.DataFrame:
    path = str(_CACHE_PATH_FORECASTS)

    if stale_scalar:
        # df includes the extra AVG_ABS_MIN_PERIOD warmup rows (see padded_period),
        # so the scalar is aligned with df already
        scalar = _compute_scalar(df)
        cache_put(key, scalar, path)
    else:
        cached = cache_get(key, path)
        # value at ts = last cached scalar at or before ts. No lookahead,
        # since every cached value was already lagged when it was computed
        scalar = (
            cached
            .reindex(columns=df.columns)
            .reindex(df.index, method="ffill")
        )

    return (df * scalar).clip(-20, 20)

def _compute_scalar(df: pd.DataFrame) -> pd.DataFrame:
    avg_abs = df.abs().expanding(min_periods=AVG_ABS_MIN_PERIOD).median()
    return 10 / avg_abs.where(avg_abs > 0)
