

from ._base import UniverseSource
from tradingsys.models.universe import CustomUniverse as CustomUniverseDefinition

class CustomUniverse(UniverseSource):
    """Wraps an explicit ticker list so it can be treated like any other Universe."""

    def __init__(self, definition: CustomUniverseDefinition) -> None:
        self._definition = definition

    def fetch_tickers(self) -> list[str]:
        return self._definition.tickers

