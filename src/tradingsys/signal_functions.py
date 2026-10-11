from __future__ import annotations

import pandas as pd 
import numpy as np 

# these functions return functions that take data as an input and data as an output 
def absolute_momentum(lookback_days: int, skip_days: int):
    def calculation(log_returns_close_data: pd.DataFrame | pd.Series):
        r_skip = log_returns_close_data.shift(skip_days)
        return (r_skip / r_skip.shift(lookback_days)) - 1
    return calculation

def donchain_breakout(lookback_days: int):
    def calculation(
            high_data: pd.DataFrame | pd.Series,
            low_data: pd.DataFrame | pd.Series,
            close_data: pd.DataFrame | pd.Series):
        rolling_high = high_data.rolling(lookback_days).max()
        rolling_low = low_data.rolling(lookback_days).min()
        rolling_mid = (rolling_high + rolling_low) / 2
        channel_width = (rolling_high - rolling_low).replace(0, np.nan)
        return (close_data - rolling_mid) / channel_width
    return calculation