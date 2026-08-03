from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .config import Config
from .state import State

logger = logging.getLogger(__name__)

class PortfolioBuilder(ABC):
    @staticmethod
    @abstractmethod
    def build_signal(cfg: Config, state: State) -> pd.DataFrame: ...

    @staticmethod
    def _make_signal_df(cfg: Config, state: State):
        trade_freq = getattr(cfg, 'trade_freq', 'W-MON')
        rebalances = (
            state.data.Close.index.to_series()
                            .resample(trade_freq)
                            .first()
                            .dropna()
        )

        return pd.DataFrame(0.0, columns=state.data.Close.columns, index=rebalances)

class Signal(ABC):
    @staticmethod
    @abstractmethod
    def signal(cfg: Config, state: State) -> None | pd.DataFrame: ...


class Filter(Signal): ...


class Indicator(Signal): ...


class SpySma200MarketRegimeFilter(Filter):
    @staticmethod
    def signal(cfg: Config, state: State) -> None:
        logger.info("Computing market regime filter (SPY vs %d-day SMA)", cfg.lookback_regime)
        try:
            spy_close = state.data.Close.SPY
            state.spy_sma_200 = spy_close.rolling(cfg.lookback_regime).mean()
            state.mrf_signal = spy_close < state.spy_sma_200
        except KeyError as e:
            raise KeyError('Expected SPY data') from e


class TsmomFilter(Filter):
    @staticmethod
    def signal(cfg: Config, state: State) -> None:
        logger.info("Computing TSMOM filter (%d-day SMA)", cfg.lookback_tsmom)
        try:
            sma = state.data.Close.rolling(cfg.lookback_tsmom).mean()
            state.tsmom_filter = state.data.Close > sma
        except KeyError as e:
            raise KeyError("Expected 'Close' in data") from e


class AverageCrossSectionalMomentumIndicator(Indicator):
    @staticmethod
    def signal(cfg: Config, state: State) -> None:
        logger.info("Computing cross-sectional momentum, lookbacks=%s, skip=%d", cfg.cross_sect_mom_lookbacks, cfg.skip_days)
        skip = cfg.skip_days
        recent = state.data.Close.shift(skip)  # ~1 month ago, per the 12-1 skip convention
        state.csm_data = pd.concat(
            [(recent / recent.shift(l - skip)) - 1 for l in cfg.cross_sect_mom_lookbacks]
        ).groupby(level=0).mean()


class FrogsInThePanIndicator(Indicator):
    @staticmethod
    def signal(cfg: Config, state: State) -> None:
        logger.info("Computing Frog-in-the-Pan indicator (%d-day lookback)", cfg.lookback_fip)
        daily_ret = state.data.Close.pct_change()
        up_days = (daily_ret > 0).rolling(cfg.lookback_fip).sum()
        down_days = (daily_ret < 0).rolling(cfg.lookback_fip).sum()
        trend_sign = np.sign(state.data.Close.pct_change(cfg.lookback_fip))
        fip = trend_sign * (down_days - up_days) / cfg.lookback_fip
        state.fip_data = -fip  # higher = smoother, more continuous trend


class SkewIndicator(Indicator):
    @staticmethod
    def signal(cfg: Config, state: State) -> None:
        logger.info("Computing skewness indicator (%d-day lookback)", cfg.lookback_skew)
        log_returns = np.log(state.data.Close).diff()
        state.skew_data = log_returns.rolling(cfg.lookback_skew).skew()


class VolatilityIndicator(Indicator):
    @staticmethod
    def signal(cfg: Config, state: State) -> None:
        logger.info("Computing volatility indicator (%d-day lookback)", cfg.lookback_volatility)
        state.vol_data = (
            np.log(state.data.Close)
            .diff()
            .rolling(cfg.lookback_volatility)
            .std()
        )
