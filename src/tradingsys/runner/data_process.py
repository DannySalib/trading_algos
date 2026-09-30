from __future__ import annotations 
from typing import Any, Callable

import pandas as pd 

from tradingsys.models.data_process import DataProcess

class DataProcessStep:
    DROP_IPO = "drop_ipo_tickers"

def construct_data_process(data_process_model: DataProcess) -> list[Callable[[pd.DataFrame], pd.DataFrame]]:
    return (
        _construct_data_process_step(name, value)
        for name, value in data_process_model
    )

def _construct_data_process_step(field_name: str, value: Any) -> Callable[[pd.DataFrame], pd.DataFrame]:
    match field_name:
        case DataProcessStep.DROP_IPO:
            return _data_process_step_drop_ipo(value)
        case _: raise NotImplementedError(f"Unkown processing step: {field_name}")

def _data_process_step_drop_ipo(drop: bool) -> Callable[[pd.DataFrame], pd.DataFrame]:
    return _drop_leading_nans if drop else lambda df: df

# The best way to handle messy data here is to drop any tickers who IPOs during your lookback. 
# This can be spotted by simply checking if the first index of the LB is NA.
# DO NOT drop tickers who have auto aujusted NANs i.e one off days with NAN values. 
# BUT ALSO do not ffill or interpolate this data! 
def _drop_leading_nans(df: pd.DataFrame):
    return df.loc[:, df.notna().idxmax().eq(df.index[0])]