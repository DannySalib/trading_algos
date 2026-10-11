from __future__ import annotations 

import pandas as pd 

from .forecast import Forecast
from ...models.signal import Signal
from ...config import CACHE_PATH
from ...cache_functions import is_stale

AVG_ABS_MIN_PERIOD = 100
_CACHE_PATH_FORECASTS = CACHE_PATH / "forecasts"
_CACHE_TTL = 60 * 60 * 24 * 30 * AVG_ABS_MIN_PERIOD

def construct_forecasts(signals: list[Signal]) -> list[Forecast]:
    try:
        return [Forecast(s) for s in signals]
    except (AttributeError, TypeError, ValueError) as e:
        raise RuntimeError("Failed to initialize forecasts") from e

def stale_forecast_scalar(forecasts: list[Forecast]):
    return is_stale(
        key=str(sorted(map(lambda f:f.name, forecasts))),
        cache_path=str(_CACHE_PATH_FORECASTS),
        ttl=_CACHE_TTL
    )

def apply_forecast_scalar(df: pd.DataFrame, stale_scalar: bool) -> pd.DataFrame:
    # TODO 
    # if we're dealing with a non-stale forecast scalar
    # df wont be aligned to cached scalar values 
    # otherwise, it'll need to be recomputed therefore it will be aligned with df 

    # scalars should be rolling and cannot peak into the future... 
    # therefore we expect 
    # ts - scalar => rolling avg(abs(forecast)) up to ts-1d
    # when stale, we load the forecast scalar so it's really 
    # ts - scalar => rolling avg(abs(forecast)) up to ts-<last cached>
    # and so we can just ffill this data 
    if stale_scalar:
        ... # recalulate 
    else:
        ... # get scalar cache and ffill 

    ...

def _scale_raw_forecast_data(df: pd.DataFrame):
    avg_abs = df.abs().expanding(min_periods=AVG_ABS_MIN_PERIOD).median()
    scalar = (10 / avg_abs.where(avg_abs > 0))
    return (df * scalar).clip(-20, 20)

def _from_position(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return data.shift(1) # remove lookahead bias in ONE place 
