import datetime as dt
from dateutil.relativedelta import relativedelta
import hashlib
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import Config

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parent / '.cache'
CACHE_PATH.mkdir(parents=True, exist_ok=True)

def cache_key(tickers: list[str]) -> str:
    """Generates a deterministic hash string for a list of tickers.

    Standard Python hash() changes every Python session/restart.
    MD5 ensures consistent filenames across program restarts.
    """
    sorted_tickers = ",".join(sorted(tickers))
    return hashlib.md5(sorted_tickers.encode("utf-8")).hexdigest()


def create_cache_path(tickers: list[str]) -> Path:
    """Returns a Path object instead of raw string."""
    return CACHE_PATH / f"{cache_key(tickers)}.parquet.gzip"


def cache_data(data: pd.DataFrame, tickers: list[str]) -> None:
    data_cache_path = create_cache_path(tickers)
    logger.info("Caching data to %s...", data_cache_path)
    data.to_parquet(
        data_cache_path,
        engine="pyarrow",
        compression='gzip',
        index=True,
    )
    logger.info("Data cached successfully.")


def read_cached_data(tickers: list[str]) -> pd.DataFrame:
    data_cache_path = create_cache_path(tickers)
    logger.info("Reading cached data from %s...", data_cache_path)

    if not data_cache_path.exists():
        raise FileNotFoundError(data_cache_path)

    return pd.read_parquet(data_cache_path, engine="pyarrow")


def _fill_small_gaps(data: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Forward-fill isolated missing trading days (holidays not caught by the
    exchange calendar, brief vendor outages, etc.) so a handful of missing
    prices don't turn into NaN returns downstream. NaN returns get silently
    dropped by pandas' skipna sum in get_portfolio_returns, which masks the
    gap as a 0% day and can produce an artificial catch-up jump whenever
    real data resumes -- forward-filling the *price* here is the correct
    place to fix that, before any return is ever computed.

    Only *interior* gaps are checked/warned on: NaN sandwiched between two
    valid observations for that column. Leading NaN (before a ticker's IPO)
    and trailing NaN (after a delisting) are excluded -- ffill can't and
    shouldn't touch those, and flagging them is just noise (every recent
    IPO in a 2,000-ticker universe would trip the warning otherwise).
    Interior gaps longer than `limit` days are left as NaN and logged:
    those usually mean a real vendor/listing issue, not a hiccup.
    """
    filled = data.ffill(limit=limit)

    notna = data.notna()
    seen_before = notna.cummax()
    seen_after = notna[::-1].cummax()[::-1]
    interior = seen_before & seen_after

    remaining_gaps = filled.isna() & interior
    if remaining_gaps.to_numpy().any():
        gap_cols = remaining_gaps.any(axis=0)
        offenders = gap_cols[gap_cols].index.tolist()
        logger.warning(
            "Interior gaps longer than %d trading days remain after "
            "forward-fill for: %s. These were left as NaN -- verify "
            "these aren't actively-held positions in the backtest.",
            limit, offenders,
        )
    return filled


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
        start = cfg.start
    else:
        last_cached = data.index[-1].normalize()

        # Last completed business day
        today = pd.Timestamp.today().normalize()
        last_business_day = today - pd.offsets.BDay(1)

        if last_cached >= last_business_day:
            logger.info("Cache is already up to date.")
            return _fill_small_gaps(data)

        start = last_cached + pd.Timedelta(days=1)

    logger.info("Downloading price data for %d tickers", len(tickers))

    new_data = yf.download(
        tickers,
        start=start,
        end=cfg.end,
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    if new_data.empty:
        if data is not None:
            logger.info("No new data available.")
            return _fill_small_gaps(data)

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
    return _fill_small_gaps(data)