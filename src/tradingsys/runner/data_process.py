from __future__ import annotations
from typing import Any, Callable

import pandas as pd
from ..models.data_process import DataProcess

class DataProcessStep:
    DROP_IPO = "drop_ipo_tickers"
    DROP_ALL_NAN = "drop_all_nan_tickers"

def construct_data_process(data_process_model: DataProcess) -> list[Callable[[pd.DataFrame], pd.DataFrame]]:
    return (
        _construct_data_process_step(name, value)
        for name, value in data_process_model
    )

def _construct_data_process_step(field_name: str, value: Any) -> Callable[[pd.DataFrame], pd.DataFrame]:
    match field_name:
        case DataProcessStep.DROP_IPO:
            return _data_process_step_drop_ipo(value)
        case DataProcessStep.DROP_ALL_NAN:
            return _data_process_step_drop_all_nan(value)
        case _: raise NotImplementedError(f"Unkown processing step: {field_name}")

def _data_process_step_drop_ipo(drop: bool) -> Callable[[pd.DataFrame], pd.DataFrame]:
    return _drop_leading_nans if drop else lambda df: df

def _data_process_step_drop_all_nan(drop: bool) -> Callable[[pd.DataFrame], pd.DataFrame]:
    return _drop_all_nan_tickers if drop else lambda df: df

# Drop any ticker whose Close is NaN on the first row of the window (IPO'd during the
# lookback, or no data at the start). Only the first row is checked, so one-off NaN days
# later in the series are kept, and nothing is ffilled or interpolated.
# An all-NaN ticker is also dropped here, since its first row is NaN.
def _drop_leading_nans(df: pd.DataFrame) -> pd.DataFrame:
    close = df.xs("Close", level="Price", axis=1)
    has_first_bar = close.iloc[0].notna()
    return _keep_tickers(df, tickers=has_first_bar.index[has_first_bar])

# Drop tickers with no Close data at all over the whole window (failed downloads, delisted
# before the period, etc.).
def _drop_all_nan_tickers(df: pd.DataFrame) -> pd.DataFrame:
    close = df.xs("Close", level="Price", axis=1)
    has_any_data = close.notna().any()
    return _keep_tickers(df, tickers=has_any_data.index[has_any_data])

def _keep_tickers(df: pd.DataFrame, tickers: pd.Index) -> pd.DataFrame:
    """Keep every column (all Price fields) belonging to the given tickers."""
    return df.loc[:, df.columns.get_level_values(0).isin(tickers)]