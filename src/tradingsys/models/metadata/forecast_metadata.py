# this will be depricated:
# forecast is TTL cache not does not follow a converage rule like price data  

# from __future__ import annotations

# from .metadata import MetaData
# from ...config import CACHE_PATH
# from datetime import date 

# _FORECAST_META_DATA_CACHE_PATH = CACHE_PATH / "forecast_metadata.json"

# class ForecastMetaData(MetaData):

#     forecast_names: list[str]
#     updated: date 
#     def save(self) -> None:
#         super().save(_FORECAST_META_DATA_CACHE_PATH)

#     @classmethod
#     def load(cls) -> ForecastMetaData:
#         return super().load(_FORECAST_META_DATA_CACHE_PATH)
