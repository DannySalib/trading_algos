from __future__ import annotations

from pydantic import BaseModel

from .period import Period
from .data_acquisition import DataAcquisition
from .universe import Universe
from .signal import Signal

class RunnerEnviron(BaseModel):
    period: Period
    data_acquisition: DataAcquisition
    universe: Universe
    signals: list[Signal]