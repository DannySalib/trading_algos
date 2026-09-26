
from __future__ import annotations

from pydantic import BaseModel
from typing import Any

from .path import FunctionPath, DataPath

class Signal(BaseModel):
    name: str
    function: FunctionPath
    data: DataPath
    other_args: dict[str, Any]
    lookback_days: int