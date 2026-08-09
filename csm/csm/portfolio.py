from __future__ import annotations

import pandas as pd

from .accessors import (
    cached_on_state
    , get_tickers_returns
    , get_spy_returns
    , get_tbill_returns
    , get_global_market_returns
)

from .signals import PortfolioBuilder, CsmPortfolioBuilder
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
    # signal_df carries 'tbill'/'glbl' regime columns alongside ticker weights,
    # so the returns frame needs matching columns for the multiply to pick them up.
    ret = pd.concat([ticker_ret, tbill_ret, glbl_ret], axis=1)
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
    pb_name = getattr(cfg, 'strategy', 'Global Equity Market')
    portfolio_builder: PortfolioBuilder = None
    match pb_name:
        case 'Global Equity Market': portfolio_builder = CsmPortfolioBuilder
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
