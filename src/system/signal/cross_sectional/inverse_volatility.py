from __future__ import annotations

import numpy as np

from system.signal.base import Signal
from system.accessors import get_tickers_returns

class InvVolSig(Signal):
    @staticmethod
    def signal(cfg, state):
        lb = getattr(cfg, 'inverse_volatility_lookback_months', 1)
        ret = get_tickers_returns(cfg, state)
        # integer window = trading days, consistent with abs_mom_lb etc.
        # shift(1): only use data through yesterday's close
        vol = ret.rolling(21 * lb).std().shift(1)
        return 1 / vol.replace(0, np.nan)

