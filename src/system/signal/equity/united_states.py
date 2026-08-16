from __future__ import annotations

from system.signal.base import Signal
from system.accessors import get_spy_price
from system.util import yearly_return

class UsEquityYtdReturns(Signal):
    @staticmethod
    def signal(cfg, state):
        spy = get_spy_price(cfg, state)
        return yearly_return(cfg, spy)