from __future__ import annotations

from pathlib import Path
import pickle as pkl

from functools import wraps

from system.data import download_data
from system.state import State
from system.util import return_
from system.data import CACHE_PATH

TICKERS_CACHE_PATH: Path = CACHE_PATH / 'ticker.pkl'
TICKERS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

def cached_on_state(attr: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(cfg, state: State, **kwargs):
            if getattr(state, attr, None) is None:
                setattr(state, attr, fn(cfg=cfg, state=state, **kwargs))
            return getattr(state, attr)
        return wrapper
    return decorator

#####################################################################################
##################### Identifying Universe and Downloading Data #####################
#####################################################################################
@cached_on_state('tickers')
def get_tickers(cfg, state):
    universe = getattr(cfg, 'universe')
    use_cached_tickers = getattr(cfg, 'use_cached_tickers', True)
    if use_cached_tickers and TICKERS_CACHE_PATH.is_file():
        with open(TICKERS_CACHE_PATH, 'rb') as f:
            return pkl.load(f)
    else:
        tickers = universe.load(cfg)
        with open(TICKERS_CACHE_PATH, 'wb') as f:
            pkl.dump(tickers, f)
        return tickers

################################
########## Price Data ##########
################################
@cached_on_state('data')
def get_ticker_data(cfg, state):
    t = get_tickers(cfg, state)
    return download_data(cfg, t)

@cached_on_state('price_spy')
def get_spy_price(cfg, state, **kwargs):
    return download_data(cfg, ['SPY']).Close.iloc[:, 0]

@cached_on_state('price_tbill')
def get_t_bill_price(cfg, state):
    return download_data(cfg, ['BIL']).Close.iloc[:, 0] / 100

@cached_on_state('price_global_market')
def get_global_market_price(cfg, state, **kwargs):
    t = getattr(cfg, 'global_market_etf_ticker', 'EFA')
    return download_data(cfg, [t]).Close.iloc[:, 0]

#######################################
########## Price Return Data ##########
#######################################
@cached_on_state('return_tbill')
def get_tbill_returns(cfg, state, **kwargs):
    tbill = get_t_bill_price(cfg, state)
    return tbill.pct_change()

@cached_on_state('return_tickers')
def get_tickers_returns(cfg, state, **kwargs):
    return return_(get_ticker_data(cfg, state).Close)

@cached_on_state('return_spy')
def get_spy_returns(cfg, state):
    return return_(get_spy_price(cfg, state))

@cached_on_state('return_global_market')
def get_global_market_returns(cfg, state, **kwargs):
    return return_(get_global_market_price(cfg, state))
