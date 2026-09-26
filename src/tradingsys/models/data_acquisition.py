from __future__ import annotations


from pydantic import BaseModel
from enum import StrEnum

class DataAcquisitionPlatform(StrEnum):
    YFINANCE = "yfinance"

class DataAcquisition(BaseModel):
    platform: DataAcquisitionPlatform