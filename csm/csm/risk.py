from __future__ import annotations

import logging

import numpy as np

from .config import Config
from .state import State

logger = logging.getLogger(__name__)


def compute_volatility_managed_returns(cfg: Config, state: State) -> None:
    """Barroso & Santa-Clara (2015), "Momentum Has Its Moments", JFE 116(1).

    Scales the momentum portfolio by target_vol / trailing realized vol, lagged
    one day to avoid look-ahead. Realized vol is estimated on invested days only
    (cash days from the regime filter are excluded) so zero-return weeks don't
    deflate the estimate right as the filter re-enters the market. Target vol
    self-calibrates to the strategy's own invested-day vol unless overridden.
    """
    raw_returns = state.portfolio_returns
    active_returns = raw_returns.where(state.invested_mask)

    min_periods = max(int(cfg.vol_management_lookback * 0.5), 1)
    realized_var_ann = (
        active_returns.pow(2)
        .rolling(cfg.vol_management_lookback, min_periods=min_periods)
        .mean()
        * cfg.trading_days
    )
    realized_vol_ann = np.sqrt(realized_var_ann).ffill()

    target_vol_ann = (
        cfg.vol_management_target
        if cfg.vol_management_target is not None
        else active_returns.std() * np.sqrt(cfg.trading_days)
    )
    state.target_vol_ann = target_vol_ann

    logger.info(
        "Applying volatility management (lookback=%dd, target=%.1f%% ann. vol, %s)"
        , cfg.vol_management_lookback, target_vol_ann * 100
        , "self-calibrated" if cfg.vol_management_target is None else "user-set"
    )

    leverage = (target_vol_ann / realized_vol_ann).shift(1)

    if cfg.vol_management_max_leverage is not None:
        leverage = leverage.clip(upper=cfg.vol_management_max_leverage)

    leverage = leverage.fillna(1.0)  # not enough invested history yet -> run unscaled

    state.leverage = leverage
    state.managed_portfolio_returns = leverage * raw_returns

    logger.info(
        "Volatility management: mean leverage=%.2f, min=%.2f, max=%.2f"
        , leverage.mean(), leverage.min(), leverage.max()
    )
