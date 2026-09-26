from __future__ import annotations

from datetime import date
from typing import Any, Self

import pandas as pd
from pandas.tseries.offsets import BDay
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from . import NotFutureDate


def _to_business_day(d: date) -> date:
    """Roll a date back to the most recent business day (Mon-Fri).

    Weekends have no market data, so any period boundary landing on a
    Saturday/Sunday rolls back to the preceding Friday. Dates that are
    already business days are returned unchanged.
    """
    return BDay().rollback(pd.Timestamp(d)).date()


class Period(BaseModel):
    model_config = ConfigDict(frozen=True)

    t0: NotFutureDate
    tf: NotFutureDate

    @model_validator(mode="before")
    @classmethod
    def _fill_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        tf = data.get("tf") or date.today()
        return {**data, "tf": tf, "t0": data.get("t0") or tf}  # copy, don't mutate

    @field_validator("t0", "tf", mode="after")
    @classmethod
    def _roll_to_business_day(cls, v: date) -> date:
        return _to_business_day(v)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.tf < self.t0:
            raise ValueError("t0 must be before or equal to tf")
        return self