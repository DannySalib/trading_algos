from __future__ import annotations

import logging
from typing import Optional, Type

import pandas as pd

from .config import Config, Param
from .state import State
from .universe import Universe
from .signals import (
    Filter,
    TsmomFilter,
    AverageCrossSectionalMomentumIndicator,
    FrogsInThePanIndicator,
    SkewIndicator,
    VolatilityIndicator,
)
from .data import download_data
from .backtest import build_signal, compute_backtest_returns
from .risk import compute_volatility_managed_returns
from .performance import compute_performance_stats, log_performance_table
from .plotting import plot_perf

logger = logging.getLogger(__name__)


def main(
        universe: Type[Universe]
        , mrf: Type[Filter]
        , param: Optional[Param] = None
) -> tuple[Config, State]:
    logger.info("Starting momentum strategy pipeline")

    cfg = Config(param=param) if param is not None else Config()
    state = State()

    state.tickers = universe.load(cfg)
    if 'SPY' not in state.tickers:
        state.tickers += ['SPY']

    state.data = download_data(cfg, state.tickers)

    mrf.signal(cfg, state)  # needs SPY still present in state.data

    state.close_spy = state.data.Close.SPY
    state.data.Close.drop(columns=['SPY'], inplace=True)

    TsmomFilter.signal(cfg, state)
    AverageCrossSectionalMomentumIndicator.signal(cfg, state)
    FrogsInThePanIndicator.signal(cfg, state)
    SkewIndicator.signal(cfg, state)
    VolatilityIndicator.signal(cfg, state)

    build_signal(cfg, state)
    compute_backtest_returns(cfg, state)
    compute_volatility_managed_returns(cfg, state)
    log_performance_table(cfg, state)
    plot_perf(cfg, state)

    logger.info("Pipeline complete")
    return cfg, state


def run_grid_search(
        universe: Type[Universe]
        , mrf: Type[Filter]
        , params: list[Param]
) -> pd.DataFrame:
    """Run the pipeline once per Param and collect summary stats.

    Example:
        grid = param_grid(top_pct=[0.03, 0.05, 0.10], csm_factor=[2, 3, 4])
        results = run_grid_search(SP500WikipediaUniverse, SpySma200MarketRegimeFilter, grid)
    """
    rows = []
    for p in params:
        cfg, state = main(universe, mrf, param=p)
        stats = compute_performance_stats(state.managed_portfolio_returns, cfg.trading_days)
        rows.append({**p.as_dict(), **stats})
    return pd.DataFrame(rows)
