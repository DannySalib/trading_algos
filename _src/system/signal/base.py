from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from system.config import Config
from system.state import State

class Signal(ABC):
    @staticmethod
    @abstractmethod
    def signal(cfg: Config, state: State) -> None | pd.DataFrame: ...
