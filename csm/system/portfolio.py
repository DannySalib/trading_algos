from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

from .accessors import (
    cached_on_state
    , get_tickers_returns
    , get_spy_returns
    , get_tbill_returns
    , get_global_market_returns
)

from system.signals import PortfolioBuilder, CsmPortfolioBuilder, GemPortfolioBuilder
from .state import State
from .util import equity, drawdown

#######################################
##########       Equity      ##########
#######################################
@cached_on_state('portfolio_returns')
def get_portfolio_returns(cfg, state, **kwargs):
    signal = get_signal_df(cfg=cfg, state=state, **kwargs)
    ticker_ret = get_tickers_returns(cfg=cfg, state=state, **kwargs)
    tbill_ret = get_tbill_returns(cfg=cfg, state=state, **kwargs).rename('tbill')
    glbl_ret = get_global_market_returns(cfg=cfg, state=state, **kwargs).rename('glbl')

    pb_name = getattr(cfg, 'strategy', 'Cross Sectional Momentum')
    match pb_name:
        case 'Cross Sectional Momentum':
            # signal_df carries 'tbill'/'glbl' regime columns alongside ticker weights,
            # so the returns frame needs matching columns for the multiply to pick them up.
            ret = pd.concat([ticker_ret, tbill_ret, glbl_ret], axis=1)
        case 'Global Equity Momentum':
            # GEM's third regime column is 'us' (SPY), not individual tickers
            us_ret = get_spy_returns(cfg=cfg, state=state, **kwargs).rename('us')
            ret = pd.concat([tbill_ret, glbl_ret, us_ret], axis=1)
        case _:
            raise RuntimeError(f'Unkown Strategy: {pb_name}')

    ret = ret.reindex(columns=signal.columns)

    # (ret * signal).sum(skipna=True) silently drops NaN contributions --
    # correct when a *zero-weight* column has no return that day, but a
    # real bug when a column you're actually holding is NaN: the position's
    # return vanishes from the sum instead of being missing, so that day
    # quietly reports as flat/0% instead of surfacing the gap. Flag it
    # instead of eating it; _fill_small_gaps() in data.py should already
    # have closed short gaps upstream, so anything left here is a real
    # data hole on a day you had capital at risk.
    held = signal.ne(0.0)
    missing_while_held = held & ret.isna()
    if missing_while_held.to_numpy().any():
        bad_dates = missing_while_held.any(axis=1)
        logger.warning(
            "Portfolio return is NaN on %d day(s) while holding a "
            "non-zero position (e.g. %s). These are being treated as 0%% "
            "return, which will understate/overstate the equity curve -- "
            "investigate the underlying price data for the held "
            "instrument(s) on those dates.",
            int(bad_dates.sum()), bad_dates[bad_dates].index[0].date(),
        )

    return (ret * signal).sum(axis=1, min_count=1)

@cached_on_state('portfolio_equity_curve')
def get_portfolio_equity(cfg, state, **kwargs):
    ret = get_portfolio_returns(cfg, state)
    return equity(ret)

@cached_on_state('benchmark_equity_curve')
def get_spy_equity(cfg, state, **kwargs):
    ret = get_spy_returns(cfg, state)
    return equity(ret)

#####################################################################################
#####################           Portfolio Building              #####################
#####################################################################################
@cached_on_state('signal_df')
def get_signal_df(cfg, state, **kwargs):
    pb_name = getattr(cfg, 'strategy', 'Cross Sectional Momentum')
    portfolio_builder: PortfolioBuilder = None
    match pb_name:
        case 'Cross Sectional Momentum': portfolio_builder = CsmPortfolioBuilder
        case 'Global Equity Momentum'  : portfolio_builder = GemPortfolioBuilder
        case _: raise RuntimeError(f'Unkown Strategy: {pb_name}')
    return portfolio_builder.build_signal(cfg, state)


#####################################################################################
#####################           Portfolio Functions             #####################
#####################################################################################
@cached_on_state('portfolio_drawdown')
def get_portfolio_drawdown(cfg, state: State, **kwargs) -> pd.Series:
    eqty = get_portfolio_equity(cfg, state)
    return drawdown(eqty)

@cached_on_state('benchmark_drawdown')
def get_spy_drawdown(cfg, state, **kwargs):
    eqty = get_spy_equity(cfg, state)
    return drawdown(eqty)

@cached_on_state('first_trade_date')
def get_first_trade_date(cfg, state, **kwargs):
    signal = get_signal_df(cfg=cfg, state=state, **kwargs)
    active = signal.abs().sum(axis=1).gt(0)
    return active[active].index[0]

# in portfolio.py, after get_portfolio_returns exists
@cached_on_state('vol_target_leverage')
def get_vol_target_leverage(cfg, state, **kwargs) -> pd.Series:
    base_ret = get_portfolio_returns(cfg, state, **kwargs)
    lb = getattr(cfg, 'vol_target_lookback_days', 126)
    target_vol = getattr(cfg, 'vol_target_annual', 0.12)
    max_lev = getattr(cfg, 'vol_target_max_leverage', 1.0)
    realized_var = (base_ret ** 2).rolling(lb).mean() * 252
    leverage = target_vol / realized_var.pow(0.5)
    return leverage.shift(1).clip(upper=max_lev).fillna(0.0)

@cached_on_state('portfolio_returns_levered')
def get_portfolio_returns_levered(cfg, state, **kwargs):
    base_ret = get_portfolio_returns(cfg, state, **kwargs)
    lev = get_vol_target_leverage(cfg, state, **kwargs)
    return base_ret * lev