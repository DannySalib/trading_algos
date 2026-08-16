from __future__ import annotations

import pandas as pd

from system.signal.base import Signal
from system.util import absolute_momentum
from system.accessors import get_t_bill_price

class AbsMomInd(Signal):
    @staticmethod
    def signal(cfg, state) -> pd.DataFrame:
        tbill_mom = absolute_momentum(cfg, get_t_bill_price(cfg, state))
        return absolute_momentum(cfg, state.data.Close).sub(tbill_mom, axis=0)