from __future__ import annotations

import numpy as np
import pandas as pd

#####################################################################################
############################## Signal-Level Calculations ############################
#####################################################################################

# skip days already shifts to only use data through t-skip -- no hindsight bias here
def absolute_momentum(cfg, data: pd.DataFrame | pd.Series):
    skip = getattr(cfg, 'skip_days', 5)  # Gary Antonacci reccomends 5 over 21
    lb = getattr(cfg, 'abs_mom_lb', 252)
    stock_prices = data.shift(skip)
    return (stock_prices / stock_prices.shift(lb)) - 1


def yearly_return(cfg, prices: pd.DataFrame | pd.Series):
    skip = getattr(cfg, 'skip_days', 5)  # Gary Antonacci reccomends 5 over 21
    current = prices.shift(skip)
    return current / current.shift(252)


def return_(data: pd.DataFrame | pd.Series):
    return data.pct_change(fill_method=None)


#####################################################################################
################################# Equity Curve Basics ################################
#####################################################################################

def equity(returns: pd.Series) -> pd.Series:
    return (returns.fillna(0) + 1).cumprod()


def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (peak - equity) / peak


def rebase(series: pd.Series, start=None) -> pd.Series:
    s = series if start is None else series.loc[start:]
    return s / s.iloc[0]


#####################################################################################
############################### Portfolio Return Assembly ############################
#####################################################################################

def portfolio_returns(signal_df: pd.DataFrame, tickers_ret: pd.DataFrame,
                       tbill_ret: pd.Series, global_ret: pd.Series) -> pd.Series:
    """Turn a weight matrix (signal_df: date x asset, already held/ffilled) into a
    realized daily portfolio return series.

    The weight on day t is the position decided as of t's close, so it earns
    the return realized on day t+1 -- shift(1) applies that lag and avoids
    look-ahead bias.
    """
    asset_ret = tickers_ret.reindex(columns=signal_df.columns.difference(['tbill', 'glbl']))
    asset_ret = asset_ret.reindex(index=signal_df.index)
    asset_ret['tbill'] = tbill_ret.reindex(signal_df.index)
    asset_ret['glbl'] = global_ret.reindex(signal_df.index)
    asset_ret = asset_ret.reindex(columns=signal_df.columns).fillna(0.0)

    weights = signal_df.shift(1).fillna(0.0)
    return (weights * asset_ret).sum(axis=1)


def first_trade_date(signal_df: pd.DataFrame) -> pd.Timestamp:
    """First date the portfolio actually holds a nonzero position, i.e. once
    lookback warm-up has passed and a real signal has fired."""
    held = signal_df.fillna(0.0).sum(axis=1)
    nonzero = held[held != 0]
    if nonzero.empty:
        raise ValueError('signal_df never holds a nonzero position')
    return nonzero.index[0]


#####################################################################################
################################ Performance Statistics ##############################
#####################################################################################

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns.where(returns < 0, 0)
    return (returns.mean() / np.sqrt(np.mean(downside ** 2))) * np.sqrt(periods_per_year)


def yearly_sharpe(returns: pd.Series, periods_per_year: int = 252) -> pd.Series:
    grouped = returns.groupby(returns.index.year)
    return (grouped.mean() / grouped.std()) * np.sqrt(periods_per_year)


def yearly_sortino(returns: pd.Series, periods_per_year: int = 252) -> pd.Series:
    downside = returns.where(returns < 0, 0)
    grouped_mean = returns.groupby(returns.index.year).mean()
    grouped_downside_rms = downside.groupby(downside.index.year).apply(
        lambda x: np.sqrt(np.mean(x ** 2))
    )
    return (grouped_mean / grouped_downside_rms) * np.sqrt(periods_per_year)


def rolling_alpha_beta(returns: pd.Series, bench_returns: pd.Series,
                        window: int = 60, periods_per_year: int = 252):
    """Rolling beta and annualized rolling alpha of `returns` vs `bench_returns`."""
    rolling_beta = returns.rolling(window).cov(bench_returns) / bench_returns.rolling(window).var()
    rolling_alpha = (
        returns.rolling(window).mean() - rolling_beta * bench_returns.rolling(window).mean()
    ) * periods_per_year
    return rolling_alpha, rolling_beta


def max_drawdown(returns: pd.Series) -> float:
    return drawdown(equity(returns)).max()