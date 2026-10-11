from __future__ import annotations

from .metadata import MetaData
from ...config import CACHE_PATH
from datetime import date 

FORECAST_META_DATA_CACHE_PATH = CACHE_PATH / "price_metadata.json"

class ForecastMetaData(MetaData):

    forecast_names: list[str]
    updated: date 
    def save(self) -> None:
        super().save(FORECAST_META_DATA_CACHE_PATH)

    @staticmethod
    def load() -> ForecastMetaData:
        return MetaData.load(FORECAST_META_DATA_CACHE_PATH)
