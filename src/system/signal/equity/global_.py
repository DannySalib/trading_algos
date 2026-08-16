from __future__ import annotations

from system.signal.base import Signal
from system.accessors import get_global_market_price
from system.util import yearly_return

class GlobalEquityYtdReturns(Signal):
    @staticmethod
    def signal(cfg, state):
        gm = get_global_market_price(cfg, state)
        return yearly_return(cfg, gm)
