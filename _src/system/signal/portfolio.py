from __future__ import annotations

from abc import abstractmethod

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import zscore

from system.signal.base import Signal
from system.accessors import get_ticker_data
from system.signal.momentum import DualMomInd
from system.signal.cross_sectional import InvVolSig, FrogInPan
from system.signal.equity import UsEquityYtdReturns, GlobalEquityYtdReturns, TbillEquityYtdReturns
from system.signal.regime import VolSpikeGate


class PortFolioSignal(Signal):
    @staticmethod
    @abstractmethod
    def build_signal(cfg, state) -> pd.DataFrame: ...

    @staticmethod
    def init_signal_df(cfg, state, extra_cols=['tbill', 'glbl']):
        data = get_ticker_data(cfg, state)
        columns = data.Close.columns.union(extra_cols)  # union keeps universe order, appends the rest
        return pd.DataFrame(np.nan, columns=columns, index=data.index)


class CsmGePortfolioSignal(PortFolioSignal):
    @staticmethod
    def signal(cfg, state) -> pd.DataFrame:
        signal_df = PortFolioSignal.init_signal_df(cfg, state)

        dm_df = DualMomInd.signal(cfg, state)
        iv_df = InvVolSig.signal(cfg, state)
        fip_df = FrogInPan.signal(cfg, state)

        ret_us_eqty_s = UsEquityYtdReturns.signal(cfg, state)
        ret_glbl_eqty_s = GlobalEquityYtdReturns.signal(cfg, state)
        ret_tbill_eqty_s = TbillEquityYtdReturns.signal(cfg, state)

        # Force all signals onto the portfolio's master date index
        ret_us_eqty_s = ret_us_eqty_s.reindex(signal_df.index)
        ret_glbl_eqty_s = ret_glbl_eqty_s.reindex(signal_df.index)
        ret_tbill_eqty_s = ret_tbill_eqty_s.reindex(signal_df.index)

        valid = ret_us_eqty_s.notna() & ret_glbl_eqty_s.notna() & ret_tbill_eqty_s.notna()

        winner_us = ret_us_eqty_s > ret_tbill_eqty_s
        winner_glbl = winner_us & (ret_glbl_eqty_s > ret_us_eqty_s)

        trade_freq = getattr(cfg, 'trade_freq', 'W-MON')
        rebalance_dates = CsmGePortfolioSignal.rebalance_dates(signal_df.index, trade_freq)

        n_basket = getattr(cfg, 'n_basket', 100)
        for d in tqdm(rebalance_dates):
            if not valid.loc[d]:
                continue  # insufficient lookback history yet -- not a signal decision, skip

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

            scores = winner_df.loc[winners, 'score']
            scores_pos = scores - scores.min() + 0.01  # epsilon avoids a zero weight for the worst-ranked winner

            conviction_weight = scores_pos * iv_df.loc[d, winners]  # score/vol, using iv_df as 1/vol
            weight = conviction_weight / conviction_weight.sum()

            row = pd.Series(0.0, index=signal_df.columns)
            row[winners] = weight
            signal_df.loc[d] = row

        # hold last rebalanced allocation until the next rebalance date
        held = signal_df.ffill().fillna(0.0)
        return CsmGePortfolioSignal._apply_vol_gate(held, cfg, state)


    @staticmethod
    def _apply_vol_gate(signal_df: pd.DataFrame, cfg, state) -> pd.DataFrame:
        """Daily override: force full cash on any day the market's short-term
        realized vol spikes relative to its trailing average, regardless of
        the weekly-held allocation."""
        gate = VolSpikeGate.signal(cfg, state).reindex(signal_df.index).fillna(False)
        gated = signal_df.copy()
        non_tbill_cols = gated.columns.difference(['tbill'])
        gated.loc[gate, non_tbill_cols] = 0.0
        gated.loc[gate, 'tbill'] = 1.0
        return gated

    @staticmethod
    def rebalance_dates(index: pd.DatetimeIndex, trade_freq: str) -> pd.DatetimeIndex:
        """First trading day present in `index` for each `trade_freq` period."""
        first_per_period = index.to_series().resample(trade_freq).first().dropna()
        return pd.DatetimeIndex(first_per_period.values)
