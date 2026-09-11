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
        skip = getattr(cfg, 'skip_days', 21)
        return (-np.sign(cum_ret) * (neg - pos)).shift(skip)


class InvVolSig(Signal):
    @staticmethod
    def signal(cfg, state):
        lb = getattr(cfg, 'inverse_volatility_lookback_months', 1)
        skip = getattr(cfg, 'skip_days', 21)
        ret = get_tickers_returns(cfg, state)
        # integer window = trading days, consistent with abs_mom_lb etc.
        # shift(1): only use data through yesterday's close
        vol = ret.rolling(21 * lb).std().shift(skip)
        return 1 / vol.replace(0, np.nan)
