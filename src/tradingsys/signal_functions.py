from __future__ import annotations

import pandas as pd 

# these functions return functions that take data as an input and data as an output 
def absolute_momentum(lookback_days: int, skip_days: int):
    def calculation(data: pd.DataFrame | pd.Series):
        r_skip = data.shift(skip_days)
        return (r_skip / r_skip.shift(lookback_days)) - 1
    return calculation
