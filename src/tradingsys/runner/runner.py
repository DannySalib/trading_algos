
from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from functools import cached_property
from datetime import timedelta
import time

logger = logging.getLogger(__name__)

from tradingsys.models import RunnerEnviron
from .data_acquisition import get_data, get_data_acquisition_client
from .universe import get_universe_tickers
from .data_process import construct_data_process
from ..models.period import Period
from ..models.signal import Signal
from ..forecast import Forecast 
from ..config import load_env

class Runner:
    def __init__(self, env: RunnerEnviron = None):
        self._env = env or load_env()
        self._forecasts = self._construct_forecasts(self._env.signals)

    @property
    def forecasts(self): return self._forecasts

    @cached_property
    def max_lookback_days(self) -> int:
        return max((s.lookback_days for s in self._env.signals), default=0)

    @cached_property
    def padded_period(self) -> Period:
        p = self._env.period
        return Period(
            t0=p.t0 - timedelta(days=self.max_lookback_days),
            tf=p.tf + timedelta(days=1)
        )

    @cached_property
    def universe_tickers(self) -> list[str]:
        return get_universe_tickers(self._env.universe)

    @cached_property
    def daq_client(self):
        return get_data_acquisition_client(self._env.data_acquisition.platform)

    @cached_property
    def ohlcv_data(self) -> pd.DataFrame:
        return self._get_processed_ohlcv_data()
    
    @cached_property
    def close_data(self) -> pd.DataFrame:
        return self.ohlcv_data.xs("Close", level="Price", axis=1)

    @cached_property
    def log_returns_close_data(self) -> pd.DataFrame:
        return np.log(self.close_data).diff()     

    ### Helpers ### 
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
        data = get_data(
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

    def _construct_forecasts(self, signals: list[Signal]) -> dict[str, Forecast]:
        try:
            return {
                s.name: Forecast(s)
                for s in signals
            }
        except (AttributeError, TypeError, ValueError) as e:
            raise RuntimeError("Failed to initialize forecasts") from e

    def _apply_forecasts(self):
        for forecast in self._forecasts.values():
            forecast.apply(
                data=getattr(self, forecast.data_need)
            )

