
import pandas as pd

# skip days already shifts to only use data through t-skip -- no hindsight bias here
def absolute_momentum(cfg, data: pd.DataFrame | pd.Series):
    skip = getattr(cfg, 'skip_days', 5) # Gary Antonacci reccomends 5 over 21
    lb = getattr(cfg, 'abs_mom_lb', 252)
    stock_prices = data.shift(skip)
    return (stock_prices / stock_prices.shift(lb)) - 1

def yearly_return(cfg, prices: pd.DataFrame | pd.Series):
    skip = getattr(cfg, 'skip_days', 5) # Gary Antonacci reccomends 5 over 21
    current = prices.shift(skip)
    return current  / current.shift(252)

def return_(data: pd.DataFrame | pd.Series):
    return data.pct_change(fill_method=None)

def equity(returns: pd.Series) -> pd.Series:
    return (returns.fillna(0)+1).cumprod()

def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (peak - equity) / peak

def rebase(series: pd.Series, start=None) -> pd.Series:
    s = series if start is None else series.loc[start:]
    return s / s.iloc[0]
