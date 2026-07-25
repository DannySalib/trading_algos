from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config
from .state import State

logger = logging.getLogger(__name__)


def compute_performance_stats(returns: pd.Series, trading_days: int) -> dict:
    mean_daily = returns.mean()
    std_daily = returns.std()

    downside = np.minimum(returns, 0.0)
    downside_dev_daily = np.sqrt((downside ** 2).mean())

    sharpe = (mean_daily / std_daily) * np.sqrt(trading_days) if std_daily > 0 else np.nan
    sortino = (mean_daily / downside_dev_daily) * np.sqrt(trading_days) if downside_dev_daily > 0 else np.nan

    growth_factor = np.exp(returns.sum())
    n_years = len(returns) / trading_days
    cagr = growth_factor ** (1 / n_years) - 1 if n_years > 0 else np.nan

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "growth_factor": growth_factor,
        "cagr": cagr,
    }


def log_performance_table(cfg: Config, state: State) -> None:
    raw_stats = compute_performance_stats(state.portfolio_returns, cfg.trading_days)
    managed_stats = compute_performance_stats(state.managed_portfolio_returns, cfg.trading_days)
    benchmark_stats = compute_performance_stats(state.benchmark_returns, cfg.trading_days)

    col_w = 14
    header = f"{'Metric':<16}{'Raw':>{col_w}}{'Vol-Managed':>{col_w}}{'Benchmark':>{col_w}}"
    sep = "-" * len(header)

    def row(label: str, key: str, pct: bool = False) -> str:
        vals = [raw_stats[key], managed_stats[key], benchmark_stats[key]]
        cells = [f"{v * 100:.2f}%" if pct else f"{v:.2f}" for v in vals]
        return f"{label:<16}" + "".join(f"{c:>{col_w}}" for c in cells)

    rows = [
        header
        , sep
        , row("Sharpe Ratio", "sharpe")
        , row("Sortino Ratio", "sortino")
        , row("Growth Factor", "growth_factor")
        , row("CAGR", "cagr", pct=True)
    ]
    logger.info(
        "Performance summary (target vol=%.1f%%, mean leverage=%.2fx):\n%s"
        , state.target_vol_ann * 100, state.leverage.mean(), "\n".join(rows)
    )
