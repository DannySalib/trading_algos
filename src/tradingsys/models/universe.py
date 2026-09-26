from __future__ import annotations

from typing import Literal
from enum import StrEnum
from pydantic import BaseModel

class UniverseName(StrEnum):
    NASDAQ = "NASDAQ"

class UniverseType(StrEnum):
    NAMED = "named"
    CUSTOM = "custom"

class NamedUniverse(BaseModel):
    type: UniverseType
    name: UniverseName

class CustomUniverse(BaseModel):
    type: UniverseType
    tickers: list[str]

class Universe(BaseModel):
    definition: NamedUniverse | CustomUniverse
