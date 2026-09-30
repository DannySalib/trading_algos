
from __future__ import annotations

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

class Runner:
    def __init__(self, env: RunnerEnviron):
        self._env = env

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
    def data(self) -> pd.DataFrame:
        logger.info(
            f"Fetching data for universe '{self._env.universe}' "
            f"over period {self.padded_period}..."
        )

        start_time = time.perf_counter()
        df = self._get_raw_ohlcv_data()
        elapsed = time.perf_counter() - start_time

        logger.info(
            f"Successfully fetched data | Shape: {df.shape} | "
            f"Took: {elapsed:.2f}s"
        )

        df = self._run_data_process(df)
        logger.info("Successfully ran data processes...")
        return df
    
    @cached_property
    def close_data(self) -> pd.DataFrame:
        return self.data.xs("Close", level="Price", axis=1)

    def _get_raw_ohlcv_data(self):
        return get_data(
            client=self.daq_client,
            tickers=self.universe_tickers,
            period=self.padded_period,
        )

    def _run_data_process(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Constructing data processor and running...")
        for process in construct_data_process(self._env.data_process):
            df = process(df)
        return df
