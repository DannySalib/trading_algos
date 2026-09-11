from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class Universe(ABC):
    @staticmethod
    @abstractmethod
    def load(cfg) -> list[str]: ...

class NasdaqUniverse(Universe):
    def load(cfg) -> list[str]:
        # Get NASDAQ stocks
        df = pd.read_csv("https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt", sep="|")[:-1]
        # drop test issues
        df = df[df['Test Issue'] == 'N']
        # drop ETFs if you only want single-name equities
        df = df[df['ETF'] == 'N']
        df = df.dropna(subset=['Symbol'])
        # drop obvious non-common-stock structures via name pattern
        junk_pattern = r'Warrant|Unit|Right|Preferred|Depositary|Acquisition Corp|SPAC|Trust'
        df = df[~df['Security Name'].str.contains(junk_pattern, case=False, na=False)]

        # Filter out tickers
        tickers  = df['Symbol'].tolist()
        batches  = [tickers[i:i+200] for i in range(0, len(tickers), 200)]
        snapshot = []
        for batch in batches:
            data = yf.download(batch, period="5d", group_by="ticker", threads=True)
            for t in batch:
                try:
                    close = data[t]['Close'].iloc[-1]
                    vol = data[t]['Volume'].mean()
                    snapshot.append({'ticker': t, 'price': close, 'avg_vol': vol})
                except Exception:
                    continue
        snap_df = pd.DataFrame(snapshot).dropna()
        snap_df['dollar_vol'] = snap_df['price'] * snap_df['avg_vol']
        snap_df  = snap_df[snap_df['price'] > 5]
        snap_df  = snap_df.sort_values('dollar_vol', ascending=False)
        n_stocks = getattr(cfg, 'nasdaq_n_stocks', 2_000)
        if n_stocks > 0:
            snap_df = snap_df.head(n_stocks)
        return snap_df['ticker'].tolist()

class GlobalEquityMomentumUniverse:
    def load(cfg) -> list[str]:
        return [] # not needed