from __future__ import annotations

import pandas as pd
import logging
from ...config import CACHE_PATH

_OHLC_CACHE = CACHE_PATH / "price_data.parquet"
logger = logging.getLogger(__name__)

from .yahoo_finance_data_acquisition import YahooFinanceDataAcquisition
from ...models.metadata import PriceMetaData
from ...models.period import Period
from ...models.data_acquisition import DataAcquisitionPlatform

from .data_acquisition_client import DataAcquisitionClient

def get_data_acquisition_client(
    platform: DataAcquisitionPlatform,
) -> DataAcquisitionClient:
    logger.info("Fetching client for data acquisition platform: %s", platform)

    match platform:
        case DataAcquisitionPlatform.YFINANCE:
            return YahooFinanceDataAcquisition()
        case _:
            raise NotImplementedError(
                f"Unrecognized Data Acquisition Platform: {platform}"
            )
    
def fetch_price_history_cached(
    client: DataAcquisitionClient,
    tickers: list[str],
    period: Period,
) -> pd.DataFrame:

    try:
        meta_data = PriceMetaData.load()
    except (FileNotFoundError, ValueError):
        meta_data = None

    cache_valid = (
        meta_data is not None
        and set(meta_data.tickers) == set(tickers)
        and meta_data.period == period
        and _OHLC_CACHE.exists()
    )

    if cache_valid:
        logger.info("Loading price data from cache: %s", _OHLC_CACHE)
        return pd.read_parquet(_OHLC_CACHE, engine="pyarrow")

    logger.info("Fetching price data for %d tickers", len(tickers))

    data = client.fetch_price_history(
        tickers,
        period.t0,
        period.tf,
    )

    data.to_parquet(_OHLC_CACHE, engine="pyarrow")
    PriceMetaData(period=period, tickers=tickers).save()

    return data