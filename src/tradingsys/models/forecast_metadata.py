from __future__ import annotations

from pydantic import BaseModel, ValidationError
from pathlib import Path
import json 

class PriceMetaData(BaseModel):
    tickers: list[str]

    def save(self, path: Path) -> None:
        try:
            with open(
                path,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(self.model_dump_json(indent=2))
        except OSError as e:
            raise OSError(
                f"Could not save price metadata to "
                f"{path}: {e}"
            ) from e

    @staticmethod
    def load(path: Path) -> PriceMetaData:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return PriceMetaData.model_validate(data)

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Price metadata cache not found: {path}"
            ) from e

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in price metadata cache "
                f"{path}: {e}"
            ) from e

        except ValidationError as e:
            raise ValueError(
                f"Invalid price metadata in {path}: {e}"
            ) from e