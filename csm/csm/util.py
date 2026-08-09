
import pandas as pd

# TODO remove hindsight bias via skip days
def absolute_momentum(cfg, data: pd.DataFrame | pd.Series):
    skip = getattr(cfg, 'skip_days', 5) # Gary Antonacci reccomends 5 over 21
    lb = getattr(cfg, 'abs_mom_lb', 252)
    stock_prices = data.shift(skip)
    return (stock_prices / stock_prices.shift(lb)) - 1

def yearly_return(cfg, data: pd.DataFrame | pd.Series):
    skip = getattr(cfg, 'skip_days', 5) # Gary Antonacci reccomends 5 over 21
    return data.shift(skip) / data.shift(skip+252)

def return_(data: pd.DataFrame | pd.Series):
    return data.pct_change(fill_method=None)

def equity(returns: pd.Series) -> pd.Series:
    return (returns.fillna(0)+1).cumprod()

def drawdown(equity: pd.Series) -> pd.Series:
    equity_cummax = equity.cummax()
    return (equity_cummax - equity) / equity

def rebase(series: pd.Series, start=None) -> pd.Series:
    s = series if start is None else series.loc[start:]
    return s / s.iloc[0]
