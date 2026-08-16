from __future__ import annotations

from system.signal.base import Signal
from system.accessors import get_t_bill_price
from system.util import yearly_return

class TbillEquityYtdReturns(Signal):
    @staticmethod
    def signal(cfg, state):
        tbill = get_t_bill_price(cfg, state)
        return yearly_return(cfg, tbill)
