from __future__ import annotations

from system.signal.base import Signal
from system.accessors import get_global_market_price, get_t_bill_price, get_spy_price
from system.util import yearly_return


class GlobalEquityYtdReturns(Signal):
    @staticmethod
    def signal(cfg, state):
        gm = get_global_market_price(cfg, state)
        return yearly_return(cfg, gm)


class TbillEquityYtdReturns(Signal):
    @staticmethod
    def signal(cfg, state):
        tbill = get_t_bill_price(cfg, state)
        return yearly_return(cfg, tbill)


class UsEquityYtdReturns(Signal):
    @staticmethod
    def signal(cfg, state):
        spy = get_spy_price(cfg, state)
        return yearly_return(cfg, spy)
