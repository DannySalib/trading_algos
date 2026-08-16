from __future__ import annotations

import pandas as pd

from system.signal.base import Signal
from system.accessors import get_spy_returns


class VolSpikeGate(Signal):
    @staticmethod
    def signal(cfg, state) -> pd.Series:
        fast_lb = getattr(cfg, 'vol_gate_fast_lb_days', 10)
        slow_lb = getattr(cfg, 'vol_gate_slow_lb_days', 60)
        threshold = getattr(cfg, 'vol_gate_threshold', 1.75)
        spy_ret = get_spy_returns(cfg, state)
        fast_vol = spy_ret.rolling(fast_lb).std()
        slow_vol = spy_ret.rolling(slow_lb).std()
        skip = getattr(cfg, 'skip_days', 21)
        # shift(1): this gate is applied daily in _apply_vol_gate, so
        # without lagging it would zero out a position using the very
        # return that position was about to earn.
        return ((fast_vol / slow_vol) > threshold).shift(skip).fillna(False)
