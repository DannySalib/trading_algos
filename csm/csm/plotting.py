from __future__ import annotations

import logging

import numpy as np
import matplotlib.pyplot as plt

from .config import Config
from .state import State

logger = logging.getLogger(__name__)
plt.style.use('dark_background')


def plot_perf(cfg: Config, state: State) -> None:
    logger.info("Plotting performance")

    portfolio_returns = state.managed_portfolio_returns
    benchmark_returns = state.benchmark_returns

    bearish_dates = state.mrf_signal[
          (state.mrf_signal.index.year >= cfg.sample_start_year)
        & (state.mrf_signal.index.year <= cfg.sample_end_year)
        & state.mrf_signal.eq(1)
    ].index

    basket_count_indicator = state.tsmom_filter.sum(axis=1)
    basket_count_indicator = basket_count_indicator[
          (basket_count_indicator.index.year >= cfg.sample_start_year)
        & (basket_count_indicator.index.year <= cfg.sample_end_year)
    ]

    rolling_covariance = portfolio_returns.rolling(cfg.lookback_benchmark).cov(benchmark_returns)
    rolling_variance = benchmark_returns.rolling(cfg.lookback_benchmark).var()
    rolling_beta = rolling_covariance / rolling_variance.replace(0, np.nan)

    rolling_alpha = (
        portfolio_returns.rolling(cfg.lookback_benchmark).mean()
        - rolling_beta
        * benchmark_returns.rolling(cfg.lookback_benchmark).mean()
    )

    portfolio_equity = np.exp(portfolio_returns.cumsum())
    benchmark_equity = np.exp(benchmark_returns.cumsum())

    portfolio_running_max = portfolio_equity.cummax()
    portfolio_drawdown    = (portfolio_equity - portfolio_running_max) / portfolio_running_max * 100

    benchmark_running_max = benchmark_equity.cummax()
    benchmark_drawdown    = (benchmark_equity - benchmark_running_max) / benchmark_running_max * 100

    logger.info(
        "Final equity: portfolio=%.3f benchmark=%.3f, max drawdown: portfolio=%.1f%% benchmark=%.1f%%"
        , portfolio_equity.iloc[-1], benchmark_equity.iloc[-1]
        , portfolio_drawdown.min(), benchmark_drawdown.min()
    )

    fig, axes = plt.subplots(
        4, 1
        , figsize=(20, 12)
        , gridspec_kw={'height_ratios': [2, 1, 1, 1]}
        , sharex=True
    )

    ymax = max(portfolio_equity.max(), benchmark_equity.max())

    ax = axes[0]
    ax.vlines(bearish_dates, ymin=0, ymax=ymax, alpha=0.1, color='white', label='Bearish')
    ax.plot(portfolio_equity, color="#42f7fa", label='Portfolio equity')
    ax.plot(benchmark_equity, color='white', label='SPY equity')
    ax.set_title('Equity Over Time')
    ax.set_ylabel('Equity (growth of $1)')

    ax_twin = ax.twinx()
    ax_twin.plot(basket_count_indicator, color="#91A39C", alpha=0.8, label='Basket count')
    ax_twin.set_ylabel('Basket count')
    ax.set_zorder(ax_twin.get_zorder() + 1)
    ax.patch.set_visible(False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
    ax.grid(True, alpha=0.3, color="#FFFFFF")

    ax = axes[1]
    ax.plot(rolling_beta, color="#FF00EE", label=f'{cfg.lookback_benchmark}d Rolling Beta')

    ax = axes[2]
    ax.plot(rolling_alpha, color="#00FF2A", label=f'{cfg.lookback_benchmark}d Rolling Alpha')

    ax = axes[3]
    ax.plot(portfolio_drawdown, color='#42f7fa', label='Portfolio Drawdown %')
    ax.fill_between(portfolio_drawdown.index, portfolio_drawdown, 0, color='#42f7fa', alpha=0.2)
    ax.plot(benchmark_drawdown, color='white', label='Benchmark Drawdown %')
    ax.fill_between(benchmark_drawdown.index, benchmark_drawdown, 0, color='white', alpha=0.2)

    for ax in axes[1:]:
        ax.grid(True, alpha=0.3, color="#FFFFFF")
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)

    plt.show()
