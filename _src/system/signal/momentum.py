from __future__ import annotations

import pandas as pd

from system.signal.base import Signal
from system.util import absolute_momentum
from system.accessors import get_t_bill_price, get_spy_price


class AbsMomInd(Signal):
    @staticmethod
    def signal(cfg, state) -> pd.DataFrame:
        tbill_mom = absolute_momentum(cfg, get_t_bill_price(cfg, state))
        return absolute_momentum(cfg, state.data.Close).sub(tbill_mom, axis=0)


# TODO relative to their category not benchmark
class RelMomInd(Signal):
    @staticmethod
    def signal(cfg, state) -> pd.DataFrame:
        spy_mom = absolute_momentum(cfg, get_spy_price(cfg, state))
        return AbsMomInd.signal(cfg, state).sub(spy_mom, axis=0)


class DualMomInd(Signal):
    @staticmethod
    def signal(cfg, state) -> pd.DataFrame:
        abs_mom = AbsMomInd.signal(cfg, state)
        rel_mom = RelMomInd.signal(cfg, state)
        return rel_mom * abs_mom.gt(0).astype(float)
