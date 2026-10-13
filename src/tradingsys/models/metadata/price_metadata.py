from __future__ import annotations

from .metadata import MetaData
from ..period import Period
from ...config import CACHE_PATH

_PRICE_META_DATA_CACHE_PATH = CACHE_PATH / "price_metadata.json"

class PriceMetaData(MetaData):
    period: Period
    tickers: list[str]

    def save(self) -> None:
        super().save(_PRICE_META_DATA_CACHE_PATH)

    @classmethod
    def load(cls) -> PriceMetaData:
        return super().load(_PRICE_META_DATA_CACHE_PATH)