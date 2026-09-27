from __future__ import annotations

import logging
import logging
import pandas as pd

from tradingsys.models.data_acquisition import DataAcquisitionPlatform
from tradingsys.models.period import Period
from tradingsys.models.price_metadata import PriceMetaData

from ._base import CACHE_PATH, DataAcquisitionClient
from .yahoo_finance_data_acquisition import YahooFinanceDataAcquisition

logger = logging.getLogger(__name__)

_OHLC_CACHE = CACHE_PATH / "price_data.parquet"

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
    
def get_data(
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