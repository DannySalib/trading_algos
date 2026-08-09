from __future__ import annotations

import datetime as dt
import hashlib
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import Config

logger = logging.getLogger(__name__)

CACHE_PATH = Path("./.cache")


def cache_key(tickers: list[str]) -> str:
    """Generates a deterministic hash string for a list of tickers.

    Standard Python hash() changes every Python session/restart.
    MD5 ensures consistent filenames across program restarts.
    """
    sorted_tickers = ",".join(sorted(tickers))
    return hashlib.md5(sorted_tickers.encode("utf-8")).hexdigest()


def create_cache_path(tickers: list[str]) -> Path:
    """Returns a Path object instead of raw string."""
    return CACHE_PATH / f"{cache_key(tickers)}.parquet"


def cache_data(data: pd.DataFrame, tickers: list[str]) -> None:
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    data_cache_path = create_cache_path(tickers)
    logger.info("Caching data to %s...", data_cache_path)
    data.to_parquet(
        data_cache_path,
        engine="pyarrow",
        compression="snappy",
        index=True,
    )
    logger.info("Data cached successfully.")


def read_cached_data(tickers: list[str]) -> pd.DataFrame:
    data_cache_path = create_cache_path(tickers)
    logger.info("Reading cached data from %s...", data_cache_path)

    if not data_cache_path.exists():
        raise FileNotFoundError(data_cache_path)

    return pd.read_parquet(data_cache_path, engine="pyarrow")


def download_data(cfg: Config, tickers: list[str]) -> pd.DataFrame:
    data: pd.DataFrame | None = None
    use_cache = getattr(cfg, 'use_cache', True)

    if use_cache:
        try:
            data = read_cached_data(tickers)
            logger.info("Loaded cached data: shape=%s", data.shape)
        except FileNotFoundError:
            logger.warning("Cache file not found. Downloading full history.")
        except Exception:
            logger.exception("Failed to read cache. Downloading full history.")

    if data is None:
        master_lookback = getattr(cfg, 'master_lookback', 365) * 10
        start = dt.datetime.today() - dt.timedelta(days=master_lookback)
    else:
        last_cached = data.index[-1].normalize()

        # Last completed business day
        today = pd.Timestamp.today().normalize()
        last_business_day = today - pd.offsets.BDay(1)

        if last_cached >= last_business_day:
            logger.info("Cache is already up to date.")
            return data

        start = last_cached + pd.Timedelta(days=1)

    logger.info("Downloading price data for %d tickers", len(tickers))

    new_data = yf.download(
        tickers,
        start=start,
        end=dt.datetime.today(),
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    if new_data.empty:
        if data is not None:
            logger.info("No new data available.")
            return data

        raise RuntimeError("No data downloaded from Yahoo Finance.")

    if data is None:
        data = new_data
    else:
        # Concatenate along time series index and drop duplicate dates
        data = (
            pd.concat([data, new_data])
            .sort_index()
            .loc[lambda df: ~df.index.duplicated(keep="last")]
        )

    cache_data(data, tickers)

    logger.info("Final data shape: %s", data.shape)
    return data