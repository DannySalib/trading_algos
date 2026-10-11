
from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from functools import cached_property
from datetime import timedelta
import time
from pathlib import Path

CACHE_PATH = Path(".cache")
logger = logging.getLogger(__name__)

from tradingsys.models import RunnerEnviron
from .data_acquisition import fetch_price_history_cached, get_data_acquisition_client
from .universe import get_universe_tickers
from .data_process import construct_data_process
from ..models.period import Period
from .forecast import (
    construct_forecasts,
    stale_forecast_scalar,
    apply_forecast_scalar,
    AVG_ABS_MIN_PERIOD
)

from ..config import load_env

class Runner:
    def __init__(self, env: RunnerEnviron = None):
        self._env = env or load_env()

        self._forecasts = construct_forecasts(self._env.signals)

        # when stale, we need to recompute the forecast scalar
        # which requres an extra `AVG_ABS_MIN_PERIOD` days of data 
        self._stale_forecast_scalar = stale_forecast_scalar(self._forecasts)

    @property
    def forecasts(self): return self._forecasts

    @property
    def forecasts_map(self):
        return {f.name:f for f in self._forecasts}

    @property
    def period(self) -> Period:
        return self._env.period

    @property
    def max_lookback_days(self) -> int:
        return max((s.lookback_days for s in self._env.signals), default=0)

    @property
    def padded_period(self) -> Period:
        p = self._env.period

        warmup_rows = self.max_lookback_days + 10 # +10 just to be safe ig 
        if self._stale_forecast_scalar:
            warmup_rows += AVG_ABS_MIN_PERIOD # we need this extra data 

        return Period(
            t0=p.t0 - timedelta(days=warmup_rows),
            tf=p.tf + timedelta(days=1),
        )

    @cached_property
    def universe_tickers(self) -> list[str]:
        return get_universe_tickers(self._env.universe)

    @property
    def daq_client(self):
        return get_data_acquisition_client(self._env.data_acquisition.platform)

    @cached_property
    def ohlcv_data(self) -> pd.DataFrame:
        return self._get_processed_ohlcv_data()

    @property
    def low_data(self) -> pd.DataFrame:
        return self.ohlcv_data.xs("Low", level="Price", axis=1)

    @property
    def high_data(self) -> pd.DataFrame:
        return self.ohlcv_data.xs("High", level="Price", axis=1)

    @property
    def close_data(self) -> pd.DataFrame:
        return self.ohlcv_data.xs("Close", level="Price", axis=1)

    @cached_property
    def log_returns_close_data(self) -> pd.DataFrame:
        return np.log(self.close_data).diff()     

    def get_unweighted_forecast_data(self):
        df = self._get_raw_forecast_data()
        return apply_forecast_scalar(df)

    ###### Helpers ######
    ### Data
    def _get_processed_ohlcv_data(self) -> pd.DataFrame:
        df = self._get_raw_ohlcv_data()
        logger.info("Constructing data processor and running...")
        for process in construct_data_process(self._env.data_process):
            df = process(df)
        logger.info("Successfully ran data processes...")
        return df

    def _get_raw_ohlcv_data(self):
        logger.info(
            f"Fetching data for universe '{self._env.universe}' "
            f"over period {self.padded_period}..."
        )
        start_time = time.perf_counter()
        data = fetch_price_history_cached(
            client=self.daq_client,
            tickers=self.universe_tickers,
            period=self.padded_period,
        )
        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Successfully fetched data | Shape: {data.shape} | "
            f"Took: {elapsed:.2f}s"
        )
        return data

    ### Forecast
    def _get_raw_forecast_data(self):
        self._create_raw_forecast_data()
        return self._concat_raw_forecast_data()
    
    def _create_raw_forecast_data(self):
        for forecast in self._forecasts:
            data = {k:getattr(self, k) for k in forecast.data_needs}
            forecast.create_raw_signal(**data)

    def _concat_raw_forecast_data(self) -> pd.DataFrame:
        return pd.concat(
            [f.raw_signal.add_prefix(f.name+'_') for f in self._forecasts],
            axis=1
        )



