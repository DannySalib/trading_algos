from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .config import Config
from .state import State

logger = logging.getLogger(__name__)

class PortfolioBuilder(ABC):
    @staticmethod
    @abstractmethod
    def build_signal(cfg: Config, state: State) -> pd.DataFrame: ...

    @staticmethod
    def _make_signal_df(cfg: Config, state: State):
        trade_freq = getattr(cfg, 'trade_freq', 'W-MON')
        rebalances = (
            state.data.Close.index.to_series()
                            .resample(trade_freq)
                            .first()
                            .dropna()
        )
        return pd.DataFrame(0.0, columns=state.data.Close.columns, index=rebalances)

class Signal(ABC):
    @staticmethod
    @abstractmethod
    def signal(cfg: Config, state: State) -> None | pd.DataFrame: ...
