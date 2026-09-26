from __future__ import annotations

from datetime import date
from typing import Annotated
from pydantic import AfterValidator

def _not_future(v: date) -> date:
    if v > date.today():
        raise ValueError(f"Date cannot be in the future: {v}")
    return v

NotFutureDate = Annotated[date, AfterValidator(_not_future)]

from .runner_env import RunnerEnviron
