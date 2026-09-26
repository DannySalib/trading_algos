from __future__ import annotations

from abc import ABC, abstractmethod
from .._base import CACHE_PATH

class UniverseSource(ABC):
    """Contract for something that can produce a list of tradeable tickers.

    Named "UniverseSource" (rather than "Universe") so it doesn't collide
    with tradingsys.models.universe.Universe, which is the config schema,
    not the runtime object. Instances are constructed with no arguments
    (or the relevant definition, as CustomUniverse does) and called via
    fetch_tickers() on an instance -- so this is a plain abstract instance
    method, not a staticmethod.
    """

    @abstractmethod
    def fetch_tickers(self) -> list[str]:
        ...
