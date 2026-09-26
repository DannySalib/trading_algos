from __future__ import annotations

from collections import defaultdict
import logging
import time

logger = logging.getLogger(__name__)

import pandas as pd
from tqdm import tqdm

from tradingsys.models.data_acquisition import DataAcquisitionPlatform
from tradingsys.models.period import Period

from ._base import DataAcquisitionClient, CACHE_PATH
from .yahoo_finance_data_acquisition import YahooFinanceDataAcquisition
from yfinance.exceptions import YFRateLimitError

_OHLC_CACHE = CACHE_PATH / "price_data.parquet"
_META_CACHE = CACHE_PATH / "price_data_meta.parquet"

def get_data_acquisition_client(platform: DataAcquisitionPlatform) -> DataAcquisitionClient:
    logger.info(f"Fetching client for data acquisition platform: {platform}")
    match platform:
        case DataAcquisitionPlatform.YFINANCE:
            return YahooFinanceDataAcquisition()
        case _:
            raise NotImplementedError(f"Unrecognized Data Acquisition Platform: {platform}")


def get_data(
    client: DataAcquisitionClient,
    tickers: list[str],
    period: Period,
) -> pd.DataFrame:
    """
    Return price data for a universe as field-major columns, e.g.::

        Open                    Close                   ...
        AAPL   MSFT   ...       AAPL   MSFT   ...        ...

    Backed by a single consolidated on-disk cache. Tickers/date-ranges
    already covered cost one metadata read and zero network calls.
    """
    logger.info(f"Getting data with client {client} over period {period} for {len(tickers)} tickers")

    meta = _read_meta()

    to_fetch: list[str] = []
    to_patch: dict[Period, list[str]] = defaultdict(list)

    for ticker in tickers:
        if meta is None or ticker not in meta.index:
            to_fetch.append(ticker)
            continue
        cached = Period(
            t0=meta.at[ticker, "first_date"].date(),
            tf=meta.at[ticker, "last_date"].date(),
        )
        missing = _find_missing(cached, period)
        if missing is not None:
            to_patch[missing].append(ticker)

    # (tickers, requested period) pairs -- kept alongside the fetched
    # frames so metadata can be bumped to "checked through" using what
    # we *asked* for, even for tickers/dates a fetch returned nothing
    # for.
    fetch_requests: list[tuple[list[str], Period]] = []
    fetched_frames: list[pd.DataFrame] = []

    if to_fetch:
        fetched_frames.append(_fetch_field_major(client, to_fetch, period.t0, period.tf))
        fetch_requests.append((to_fetch, period))

    # Tickers sharing an identical gap (the common case: "just today")
    # are fetched together, so this is typically one call, not one per
    # ticker.
    for missing, group in to_patch.items():
        fetched_frames.append(_fetch_field_major(client, group, missing.t0, missing.tf))
        fetch_requests.append((group, missing))

    if fetch_requests:
        cache = _read_cache()
        for frame in fetched_frames:
            if frame.empty:
                continue
            cache = _merge_frame(frame, cache)
        if cache is not None:
            cache = cache.sort_index().sort_index(axis=1)
            _save_cache(cache)
        _save_meta(_bump_meta(meta, fetch_requests))
    else:
        cache = _read_cache()

    if cache is None:
        raise RuntimeError("No cached or fetched price data available")

    logger.info("Got the data... slicing...")
    return _slice(cache, period, tickers)


def _fetch_field_major(client: DataAcquisitionClient, tickers: list[str], t0, tf) -> pd.DataFrame:
    raw = _chunked_fetch(client, tickers, t0, tf)
    if raw.empty:
        return raw
    raw.index = _normalize_index(raw.index)
    return raw.swaplevel(axis=1).sort_index(axis=1)


def _chunked_fetch(client, tickers: list[str], t0, tf, chunk_size: int = 200) -> pd.DataFrame:
    frames = []
    for i in tqdm(range(0, len(tickers), chunk_size), desc="Fetching price data"):
        chunk = tickers[i:i + chunk_size]
        for attempt in range(3):
            try:
                data = client.fetch_price_history(chunk, t0, tf)
                frames.append(_ensure_ticker_major(data, chunk))
                break
            except YFRateLimitError:
                wait = 10 * (attempt + 1)
                logger.warning(f"Rate limited on chunk {i}-{i+chunk_size}, waiting {wait}s")
                time.sleep(wait)
        time.sleep(1)  # small gap between chunks regardless
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def _ensure_ticker_major(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalize a fetch result to (ticker, field) MultiIndex columns.

    yfinance flattens columns to just the field names for the single-
    ticker edge case even with group_by="ticker", so a chunk of exactly
    one ticker needs this restored explicitly.
    """
    if isinstance(data.columns, pd.MultiIndex):
        return data
    if len(tickers) == 1:
        return pd.concat({tickers[0]: data}, axis=1)
    raise ValueError(f"Expected MultiIndex columns for tickers={tickers}, got flat columns")


def _merge_frame(new: pd.DataFrame, existing: pd.DataFrame | None) -> pd.DataFrame:
    """Outer-join `new` onto `existing`, with `new`'s values winning on overlap.

    `DataFrame.combine_first`/`.update` are column-at-a-time under the
    hood, which is fine for a handful of columns but becomes the
    dominant cost -- tens of seconds -- once the frame spans a whole
    universe (thousands of tickers x 5 fields). Reindexing both frames
    onto a shared axis and doing one vectorized `.where` does the same
    outer-join-with-new-winning semantics in a fraction of the time.
    """
    if existing is None:
        return new

    idx = existing.index.union(new.index)
    cols = existing.columns.union(new.columns)
    existing_r = existing.reindex(index=idx, columns=cols)
    new_r = new.reindex(index=idx, columns=cols)
    return new_r.where(new_r.notna(), existing_r)


def _find_missing(cached: Period, requested: Period) -> Period | None:
    start = pd.Timestamp(requested.t0)
    end = pd.Timestamp(requested.tf)
    cached_start = pd.Timestamp(cached.t0)
    cached_end = pd.Timestamp(cached.tf)

    # Requested period is completely contained in what's cached.
    if cached_start <= start and cached_end >= end:
        return None

    # Missing data before the cached range.
    if start < cached_start:
        return Period(t0=start, tf=min(end, cached_start))

    # Missing data after the cached range.
    return Period(t0=max(start, cached_end), tf=end)


def _bump_meta(meta: pd.DataFrame | None, requests: list[tuple[list[str], Period]]) -> pd.DataFrame:
    rows: dict[str, dict[str, pd.Timestamp]] = {} if meta is None else meta.to_dict(orient="index")

    for group_tickers, requested in requests:
        t0 = pd.Timestamp(requested.t0)
        tf = pd.Timestamp(requested.tf)
        for ticker in group_tickers:
            existing = rows.get(ticker)
            if existing is None:
                rows[ticker] = {"first_date": t0, "last_date": tf}
            else:
                rows[ticker] = {
                    "first_date": min(existing["first_date"], t0),
                    "last_date": max(existing["last_date"], tf),
                }

    return pd.DataFrame.from_dict(rows, orient="index")


def _slice(cache: pd.DataFrame, period: Period, tickers: list[str]) -> pd.DataFrame:
    sliced = cache.loc[pd.Timestamp(period.t0):pd.Timestamp(period.tf)]
    fields = sliced.columns.get_level_values(0).unique()
    cols = pd.MultiIndex.from_product([fields, tickers])
    return sliced.reindex(columns=cols)


def _save_cache(data: pd.DataFrame) -> None:
    if data.empty:
        return
    _OHLC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(_OHLC_CACHE)


def _read_cache() -> pd.DataFrame | None:
    if not _OHLC_CACHE.exists():
        return None
    try:
        return pd.read_parquet(_OHLC_CACHE)
    except Exception:
        logger.warning(f"Cache at {_OHLC_CACHE} is unreadable, refetching")
        _OHLC_CACHE.unlink(missing_ok=True)
        return None


def _save_meta(meta: pd.DataFrame) -> None:
    _META_CACHE.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(_META_CACHE)


def _read_meta() -> pd.DataFrame | None:
    if not _META_CACHE.exists():
        return None
    try:
        return pd.read_parquet(_META_CACHE)
    except Exception:
        logger.warning(f"Metadata cache at {_META_CACHE} is unreadable, rebuilding")
        _META_CACHE.unlink(missing_ok=True)
        return None


def _normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(index)

    if index.tz is not None:
        index = index.tz_localize(None)

    return index.normalize()
