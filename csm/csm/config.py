from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass, field, replace, asdict
from typing import Optional


@dataclass(frozen=True)
class Param:
    """Tunable hyperparameters -- what a grid search sweeps over."""
    @property
    def lookback_csm(self) -> int:
        return max(self.cross_sect_mom_lookbacks)

    def as_dict(self) -> dict:
        return asdict(self)

    def with_overrides(self, **kwargs) -> "Param":
        return replace(self, **kwargs)


def param_grid(**value_lists) -> list[Param]:
    """Build every combination of field=[values,...] as a list of Param."""
    keys = list(value_lists.keys())
    combos = itertools.product(*value_lists.values())
    return [Param(**dict(zip(keys, combo))) for combo in combos]


@dataclass(frozen=True)
class Config:
    """Infra settings, never swept. Unknown attrs fall through to `param`."""
    param: Param = field(default_factory=Param)

    @property
    def sample_end_year(self) -> int:
        return dt.date.today().year

    @property
    def sample_start_year(self) -> int:
        return self.sample_end_year - self.sample_lookback_years

    def with_param(self, param: Param) -> "Config":
        return replace(self, param=param)

    def __getattr__(self, name: str):
        try:
            return getattr(self.param, name)
        except AttributeError:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
