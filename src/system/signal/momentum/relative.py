from __future__ import annotations

import pandas as pd

from system.signal.base import Signal
from system.util import absolute_momentum
from system.accessors import get_spy_price

from .absolute import AbsMomInd

# TODO relative to their category not benchmark
class RelMomInd(Signal):
    @staticmethod
    def signal(cfg, state) -> pd.DataFrame:
        spy_mom = absolute_momentum(cfg, get_spy_price(cfg, state))
        return AbsMomInd.signal(cfg, state).sub(spy_mom, axis=0)