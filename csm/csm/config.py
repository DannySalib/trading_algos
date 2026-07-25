from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass, field, replace, asdict
from typing import Optional


@dataclass(frozen=True)
class Param:
    """Tunable hyperparameters -- what a grid search sweeps over."""

    lookback_regime: int = 200
    lookback_tsmom: int = 200

    cross_sect_mom_lookbacks: list = field(default_factory=lambda: [60, 120, 252])
    skip_days: int = 21

    lookback_fip: int = 252
    lookback_skew: int = 90
    lookback_volatility: int = 126

    trade_freq: str = 'W-MON'

    top_pct     : float = 0.05
    csm_factor  : int   = 3
    fip_factor  : int   = 1
    skew_factor : int   = 2

    # Barroso & Santa-Clara (2015), "Momentum Has Its Moments", JFE 116(1).
    # target=None self-calibrates to the strategy's own invested-day vol.
    vol_management_lookback: int = 126
    vol_management_target: Optional[float] = None
    vol_management_max_leverage: Optional[float] = 2.0

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

    sp500_wiki_url: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    russell2000_url : str = "https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares&locale=en_US&portfolioId=239710&component=fundDownload&userType=individual",


    table_index: int  = 0
    ticker_col : str  = "Symbol"
    headers    : dict = field(default_factory=lambda: {"User-Agent": "Mozilla/5.0"})

    use_cache: bool = True

    n_years: int = 10
    master_lookback: int = 365 * n_years

    lookback_benchmark: int = 63
    trading_days: int = 252

    sample_lookback_years: int = 8

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
