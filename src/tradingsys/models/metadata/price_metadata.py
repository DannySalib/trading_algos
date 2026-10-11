from __future__ import annotations

from .metadata import MetaData
from ..period import Period
from ...config import CACHE_PATH

PRICE_META_DATA_CACHE_PATH = CACHE_PATH / "forecast_metadata.json"

class PriceMetaData(MetaData):
    period: Period
    tickers: list[str]
    
    def save(self) -> None:
        super().save(PRICE_META_DATA_CACHE_PATH)

    @staticmethod
    def load() -> PriceMetaData:
        return MetaData.load(PRICE_META_DATA_CACHE_PATH)
