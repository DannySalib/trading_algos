from __future__ import annotations

import pandas as pd
import numpy as np

from system.signal.base import Signal
from system.accessors import get_ticker_data

from abc import abstractmethod

class PortFolioSignal(Signal):
    @staticmethod
    @abstractmethod
    def build_signal(cfg, state) -> pd.DataFrame: ...

    @staticmethod
    def init_signal_df(cfg, state, extra_cols=['tbill', 'glbl']):
        data = get_ticker_data(cfg, state)
        columns = data.Close.columns.union(extra_cols)  # union keeps universe order, appends the rest
        return pd.DataFrame(np.nan, columns=columns, index=data.index)
