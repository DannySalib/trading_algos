from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd
from .._base import CACHE_PATH


class DataAcquisitionClient(ABC):
    """Contract for a market-data backend.

    Named "Client" (rather than "DataAcquisition") so it doesn't collide
    with tradingsys.models.data_acquisition.DataAcquisition, which is the
    config schema, not the runtime object.
    """

    @abstractmethod
    def fetch_price_history(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Return historical OHLCV price data for the given tickers and date range."""
        ...
