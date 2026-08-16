from __future__ import annotations

import numpy as np

from system.signal.base import Signal
from system.accessors import get_tickers_returns

class FrogInPan(Signal):
    @staticmethod
    def signal(cfg, state):
        ret = get_tickers_returns(cfg, state)
        lb = getattr(cfg, 'frog_in_pan_lookback_months', 1)
        window = 21 * lb
        pos = ret.gt(0).rolling(window).mean()
        neg = ret.lt(0).rolling(window).mean()
        cum_ret = ret.rolling(window).apply(
            lambda x: np.prod(1 + x) - 1,
            raw=True,
        )
        # shift(1): rolling() at row d includes today's return (unknown
        # until today's close) -- lag by 1 so the signal used to trade
        # day d only reflects data through d-1's close.
        return (-np.sign(cum_ret) * (neg - pos)).shift(1)