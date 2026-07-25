from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class State:
    """Empty/None at the start, filled in as the pipeline runs."""

    tickers: list = field(default_factory=list)

    data: Optional[pd.DataFrame] = None
    close_spy: Optional[pd.Series] = None

    spy_sma_200: Optional[pd.Series] = None
    mrf_signal: Optional[pd.Series] = None

    tsmom_filter: Optional[pd.DataFrame] = None

    csm_data: Optional[pd.DataFrame] = None
    fip_data: Optional[pd.DataFrame] = None
    skew_data: Optional[pd.DataFrame] = None
    vol_data: Optional[pd.DataFrame] = None

    signal_df: Optional[pd.DataFrame] = None
    top_n: Optional[int] = None

    portfolio_returns: Optional[pd.Series] = None
    benchmark_returns: Optional[pd.Series] = None
    invested_mask: Optional[pd.Series] = None

    leverage: Optional[pd.Series] = None
    target_vol_ann: Optional[float] = None
    managed_portfolio_returns: Optional[pd.Series] = None
