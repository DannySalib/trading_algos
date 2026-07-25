from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config
from .state import State

logger = logging.getLogger(__name__)


def normalize(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def _build_signal(d: pd.Timestamp, cfg: Config, state: State) -> None:
    signal_df    = state.signal_df
    mrf_signal   = state.mrf_signal
    tsmom_filter = state.tsmom_filter
    csm_data     = state.csm_data
    fip_data     = state.fip_data
    skew_data    = state.skew_data
    vol_data     = state.vol_data
    top_n        = state.top_n

    if mrf_signal.loc[d]:
        signal_df.loc[d] = 0.0
        return

    basket = signal_df.columns[tsmom_filter.loc[d]]
    csm = (
        normalize(csm_data.loc[d][basket][lambda x: x > 0])
        if not basket.empty else pd.Series(dtype=float)
    )

    if basket.empty or csm.empty:
        signal_df.loc[d] = signal_df.shift(1).loc[d]  # carry forward, already sums to 1
        return

    winners = csm.index
    fip = normalize(fip_data.loc[d, winners])
    skew = normalize(skew_data.loc[d, winners])

    score = (
          (cfg.csm_factor  * csm)
        + (cfg.fip_factor  * fip)
        + (cfg.skew_factor * skew)
    )
    top_picks = score.sort_values(ascending=False).head(top_n)

    vol = vol_data.loc[d, top_picks.index]
    inv_vol = 1 / vol
    stock_weights = inv_vol / inv_vol.sum()  # inverse-vol weighting, sums to 1

    signal_df.loc[d] = 0.0
    signal_df.loc[d, stock_weights.index] = stock_weights


def build_signal(cfg: Config, state: State) -> None:
    rebalances = (
        state.data.Close.index.to_series()
                        .resample(cfg.trade_freq)
                        .first()
                        .dropna()
    )

    signal_df = pd.DataFrame(0.0, columns=state.data.Close.columns, index=rebalances)

    state.signal_df = signal_df
    state.top_n = int(len(signal_df.columns) * cfg.top_pct)

    logger.info("Building signals for %d rebalance dates (top_n=%d)", len(signal_df.index), state.top_n)

    for d in signal_df.index:
        _build_signal(d, cfg, state)

    logger.info("Signal construction complete")


def compute_backtest_returns(cfg: Config, state: State) -> None:
    logger.info("Computing backtest returns (sample %d-%d)", cfg.sample_start_year, cfg.sample_end_year)

    df_returns = state.data.Close.apply(lambda x: np.log(x).diff().shift(-1), axis=0)

    df_weights = state.signal_df.reindex(df_returns.index).ffill().fillna(0)
    invested_mask = df_weights.sum(axis=1) > 0

    portfolio_returns = (df_returns * df_weights).ffill().sum(axis=1)
    benchmark_returns = np.log(state.close_spy).diff().shift(-1)

    portfolio_returns = portfolio_returns[
          (portfolio_returns.index.year >= cfg.sample_start_year)
        & (portfolio_returns.index.year <= cfg.sample_end_year)
    ]
    benchmark_returns = benchmark_returns[
          (benchmark_returns.index.year >= cfg.sample_start_year)
        & (benchmark_returns.index.year <= cfg.sample_end_year)
    ]
    invested_mask = invested_mask.reindex(portfolio_returns.index).fillna(False)

    state.portfolio_returns = portfolio_returns
    state.benchmark_returns = benchmark_returns
    state.invested_mask = invested_mask

    logger.info(
        "Backtest returns computed: %d observations (%d invested, %d in cash)"
        , len(portfolio_returns), int(invested_mask.sum()), int((~invested_mask).sum())
    )
