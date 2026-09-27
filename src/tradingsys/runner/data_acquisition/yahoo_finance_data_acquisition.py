from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf
# import requests_cache

from ._base import DataAcquisitionClient

# CACHE_PATH_YF_SESSION = CACHE_PATH / "yf_session"

class YahooFinanceDataAcquisition(DataAcquisitionClient):
    def __init__(self):
        super().__init__()
        # self._session = requests_cache.CachedSession(CACHE_PATH_YF_SESSION, expire_after=3600)

    """Fetches historical OHLCV data from Yahoo Finance via yfinance."""
    def fetch_price_history(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        # yfinance treats `end` as exclusive, so a same-day request like
        # start == end (today's bar) would otherwise always come back
        # empty. Push the boundary out by one day to make `end` inclusive.
        return yf.download(
            tickers,
            start=start,
            end=end + timedelta(days=1),
            group_by="ticker",
            threads=True,
            auto_adjust=True,
            # progress=False,
            # session=self._session
        )