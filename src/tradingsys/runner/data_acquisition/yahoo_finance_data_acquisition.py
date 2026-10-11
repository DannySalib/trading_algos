from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from .data_acquisition_client import DataAcquisitionClient

class YahooFinanceDataAcquisition(DataAcquisitionClient):
    def __init__(self):
        super().__init__()

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
        )