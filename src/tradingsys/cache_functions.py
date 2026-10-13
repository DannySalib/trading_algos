
from typing import Callable, Any
import shelve
import time

def cache_put(key: str, value: Any, cache_path: str) -> None:
    with shelve.open(cache_path) as cache:
        cache[key] = {"timestamp": time.time(), "value": value}

def cache_get(key: str, cache_path: str) -> Any:
    with shelve.open(cache_path) as cache:
        entry = cache.get(key)
        if entry is None:
            raise KeyError(f"No cache entry for {key}")
        return entry["value"]

def cached_fetch(
    key: str,
    fetch: Callable[[], Any],
    cache_path: str,
    ttl: float,
) -> Any:
    with shelve.open(cache_path) as cache:
        entry = cache.get(key)

        if (
            entry is not None
            and time.time() - entry["timestamp"] < ttl
        ):
            return entry["value"]

        value = fetch()

        cache[key] = {
            "timestamp": time.time(),
            "value": value,
        }

        return value
    
def is_stale(key: str, cache_path: str, ttl: float) -> bool:
    with shelve.open(cache_path) as cache:
        entry = cache.get(key)

        return (
            entry is None
            or time.time() - entry["timestamp"] >= ttl
        )