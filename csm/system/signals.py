from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import zscore

from .config import Config
from .state import State
from .accessors import (
    get_t_bill_price, get_spy_price, get_global_market_price
    , get_tickers_returns, get_spy_returns
    , get_ticker_data
)
from .util import (
    yearly_return
    , absolute_momentum
)

logger = logging.getLogger(__name__)

class PortfolioBuilder(ABC):
    @staticmethod
    @abstractmethod
    def build_signal(cfg: Config, state) -> pd.DataFrame: ...

    @staticmethod
    def _make_signal_df(cfg: Config, state: State):
        data = get_ticker_data(cfg, state)
        return pd.DataFrame(np.nan, columns=data.Close.columns, index=data.index)


class Signal(ABC):
    @staticmethod
    @abstractmethod
    def signal(cfg: Config, state: State) -> None | pd.DataFrame: ...

##########################################################################
#####################     My Portfolio Builder       #####################
##########################################################################
class CsmPortfolioBuilder(PortfolioBuilder):
    @staticmethod
    def build_signal(cfg: Config, state: State) -> pd.DataFrame:
        signal_df = PortfolioBuilder._make_signal_df(cfg, state)

        dm_df = DualMomInd.signal(cfg, state)
        iv_df = InvVolSig.signal(cfg, state)
        fip_df = FrogInPan.signal(cfg, state)
        ret_us_eqty_s = UsEquityYtdReturns.signal(cfg, state)
        ret_glbl_eqty_s = GlobalEquityYtdReturns.signal(cfg, state)
        ret_tbill_eqty_s = TbillEquityYtdReturns.signal(cfg, state)

        winner_us = ret_us_eqty_s > ret_tbill_eqty_s
        winner_glbl = winner_us & (ret_glbl_eqty_s > ret_us_eqty_s)

        trade_freq = getattr(cfg, 'trade_freq', 'W-MON')
        rebalance_dates = CsmPortfolioBuilder._rebalance_dates(signal_df.index, trade_freq)

        n_basket = getattr(cfg, 'n_basket', 100)
        for d in tqdm(rebalance_dates):
            if winner_glbl.loc[d]:
                row = pd.Series(0.0, index=signal_df.columns)
                row['glbl'] = 1.0
                signal_df.loc[d] = row
                continue

            if not winner_us.loc[d]:
                row = pd.Series(0.0, index=signal_df.columns)
                row['tbill'] = 1.0
                signal_df.loc[d] = row
                continue

            winner_df = dm_df.loc[d][lambda x: x.gt(0)].to_frame('dm')
            if winner_df.empty: continue  # no winners -> hold prior allocation via ffill

            winner_df['fip'] = fip_df.loc[d, winner_df.index]
            winner_df = winner_df.dropna()
            if winner_df.empty: continue  # same

            winner_df = winner_df.apply(zscore)
            winner_df['score'] = winner_df.sum(axis=1)
            winners = winner_df.nlargest(n_basket, 'score').index

            weight = iv_df.loc[d, winners] / iv_df.loc[d, winners].sum()
            row = pd.Series(0.0, index=signal_df.columns)
            row[winners] = weight
            signal_df.loc[d] = row

        # hold last rebalanced allocation until the next rebalance date
        held = signal_df.ffill().fillna(0.0)
        return CsmPortfolioBuilder._apply_vol_gate(held, cfg, state)

    @staticmethod
    def _rebalance_dates(index: pd.DatetimeIndex, trade_freq: str) -> pd.DatetimeIndex:
        """First trading day present in `index` for each `trade_freq` period."""
        first_per_period = index.to_series().resample(trade_freq).first().dropna()
        return pd.DatetimeIndex(first_per_period.values)

    @staticmethod
    def _apply_vol_gate(signal_df: pd.DataFrame, cfg: Config, state: State) -> pd.DataFrame:
        """Daily override: force full cash on any day the market's short-term
        realized vol spikes relative to its trailing average, regardless of
        the weekly-held allocation."""
        gate = VolSpikeGate.signal(cfg, state).reindex(signal_df.index).fillna(False)
        gated = signal_df.copy()
        non_tbill_cols = gated.columns.difference(['tbill'])
        gated.loc[gate, non_tbill_cols] = 0.0
        gated.loc[gate, 'tbill'] = 1.0
        return gated

##########################################################################
##################### Global Equity Momentum Signals #####################
##########################################################################
class TbillEquityYtdReturns(Signal):
    @staticmethod
    def signal(cfg, state):
        tbill = get_t_bill_price(cfg, state)
        return yearly_return(cfg, tbill)

class UsEquityYtdReturns(Signal):
    @staticmethod
    def signal(cfg, state):
        spy = get_spy_price(cfg, state)
        return yearly_return(cfg, spy)

class GlobalEquityYtdReturns(Signal):
    @staticmethod
    def signal(cfg, state):
        gm = get_global_market_price(cfg, state)
        return yearly_return(cfg, gm)

##########################################################################
#####################      Dual Momentum Signals     #####################
##########################################################################
class DualMomInd(Signal):
    @staticmethod
    def signal(cfg, state) -> pd.DataFrame:
        abs_mom = AbsMomInd.signal(cfg, state)
        rel_mom = RelMomInd.signal(cfg, state)
        return rel_mom * abs_mom.gt(0).astype(float)

# TODO relative to their category not benchmark
class RelMomInd(Signal):
    @staticmethod
    def signal(cfg, state) -> pd.DataFrame:
        spy_mom = absolute_momentum(cfg, get_spy_price(cfg, state))
        return AbsMomInd.signal(cfg, state).sub(spy_mom, axis=0)

class AbsMomInd(Signal):
    @staticmethod
    def signal(cfg, state) -> pd.DataFrame:
        tbill_mom = absolute_momentum(cfg, get_t_bill_price(cfg, state))
        return absolute_momentum(cfg, state.data.Close).sub(tbill_mom, axis=0)

# # this sucks
# class SpySmaMrf(Signal):
#     @staticmethod
#     def signal(cfg, state):
#         lb = getattr(cfg, 'spy_sma_lb', 252)
#         return state.benchmark <= state.benchmark.rolling(lb).mean()

##########################################################################
#####################    Momentum Related Signals    #####################
##########################################################################
class InvVolSig(Signal):
    @staticmethod
    def signal(cfg, state):
        lb = getattr(cfg, 'inverse_volatility_lookback', 21)
        ret = get_tickers_returns(cfg, state)
        vol = ret.rolling(lb).std().shift(1)
        return 1 / vol.replace(0, np.nan)

class FrogInPan(Signal):
    @staticmethod
    def signal(cfg, state):
        ret = get_tickers_returns(cfg, state)
        lb = getattr(cfg, "frog_in_pan_lb", 21)
        pos = ret.gt(0).rolling(lb).mean()
        neg = ret.lt(0).rolling(lb).mean()
        cum_ret = ret.rolling(lb).apply(
            lambda x: np.prod(1 + x) - 1,
            raw=True,
        )
        return -np.sign(cum_ret) * (neg - pos)

class VolSpikeGate(Signal):
    @staticmethod
    def signal(cfg, state) -> pd.Series:
        fast_lb = getattr(cfg, 'vol_gate_fast_lb', 10)
        slow_lb = getattr(cfg, 'vol_gate_slow_lb', 60)
        threshold = getattr(cfg, 'vol_gate_threshold', 1.75)

        spy_ret = get_spy_returns(cfg, state)
        fast_vol = spy_ret.rolling(fast_lb).std()
        slow_vol = spy_ret.rolling(slow_lb).std()
        return (fast_vol / slow_vol) > threshold

