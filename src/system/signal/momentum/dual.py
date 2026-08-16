

import pandas as pd

from system.signal.base import Signal
from .absolute import AbsMomInd
from .relative import RelMomInd

class DualMomInd(Signal):
    @staticmethod
    def signal(cfg, state) -> pd.DataFrame:
        abs_mom = AbsMomInd.signal(cfg, state)
        rel_mom = RelMomInd.signal(cfg, state)
        return rel_mom * abs_mom.gt(0).astype(float)