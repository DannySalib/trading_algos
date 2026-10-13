from __future__ import annotations

from tradingsys.models.universe import (
    NamedUniverse,
    UniverseName,
    UniverseType,
    Universe as UniverseModel,
)

from .universe_source import UniverseSource
from .nasdaq_universe import NasdaqUniverse
from .custom_universe import CustomUniverse
from ...config import CACHE_PATH
from ...cache_functions import cached_fetch

_CACHE_PATH_UNIVERSE = CACHE_PATH / "universes"
_CACHE_PATH_UNIVERSE.parent.mkdir(parents=True, exist_ok=True)

_CACHE_TTL = 60 * 60 * 24 * 30 * 6

_NAMED_UNIVERSES: dict[UniverseName, type[UniverseSource]] = {
    UniverseName.NASDAQ: NasdaqUniverse,
}

def get_universe_tickers(universe: UniverseModel) -> list[str]:
    definition = universe.definition

    match definition.type:
        case UniverseType.CUSTOM:
            return CustomUniverse(definition).fetch_tickers()

        case UniverseType.NAMED:
            return _get_cached_named(definition)

        case _:
            raise NotImplementedError(
                f"Unrecognized universe type: {definition.type}"
            )

def _get_cached_named(definition: NamedUniverse) -> list[str]:
    return cached_fetch(
        key=definition.name.value,
        fetch=_resolve_named_universe(definition).fetch_tickers,
        cache_path=str(_CACHE_PATH_UNIVERSE),
        ttl=_CACHE_TTL
    )

def _resolve_named_universe(definition: NamedUniverse) -> UniverseSource:
    try:
        universe_cls = _NAMED_UNIVERSES[definition.name]
    except KeyError:
        raise NotImplementedError(
            f"Unrecognized universe name: {definition.name}"
        ) from None

    return universe_cls()