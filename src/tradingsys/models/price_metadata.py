from __future__ import annotations

import json 
from pydantic import BaseModel, ValidationError

from .period import Period 
from ..config import CACHE_PATH

PRICE_META_DATA_CACHE_PATH = CACHE_PATH / "price_metadata.json"

class PriceMetaData(BaseModel):
    period: Period
    tickers: list[str]

    def save(self) -> None:
        try:
            with open(
                PRICE_META_DATA_CACHE_PATH,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(self.model_dump_json(indent=2))
        except OSError as e:
            raise OSError(
                f"Could not save price metadata to "
                f"{PRICE_META_DATA_CACHE_PATH}: {e}"
            ) from e

    @staticmethod
    def load() -> PriceMetaData:
        try:
            with open(PRICE_META_DATA_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            return PriceMetaData.model_validate(data)

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Price metadata cache not found: {PRICE_META_DATA_CACHE_PATH}"
            ) from e

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in price metadata cache "
                f"{PRICE_META_DATA_CACHE_PATH}: {e}"
            ) from e

        except ValidationError as e:
            raise ValueError(
                f"Invalid price metadata in {PRICE_META_DATA_CACHE_PATH}: {e}"
            ) from e