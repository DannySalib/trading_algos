import os
from datetime import datetime, timedelta
from io import StringIO
import requests
import yfinance as yf
import pandas as pd 

CACHE = 'sp500_tickers_close.pq.gzip'
STALE_AFTER_DAYS = 1

def build_signal_df(df: pd.DataFrame
                    , drawdown_signal_threshold             : float = -0.10
                    , vix_signal_threshold                  : int   = 20
                    , breadth_thrust_signal_threshold       : float = 0.30
                    , fear_zscore_signal_threshold          : float = -1.5
                    , rsi_signal_threshold                  : int   = 35
                    ) -> pd.DataFrame:

    df.index = df.index.tz_convert('US/Eastern')

    try:
        signal_df = df[['Close']].copy()
    except KeyError:
        raise KeyError('Must contain closing price time series data')

    # === SPY Drawdown signal
    signal_df['high_52w'] = signal_df['Close'].rolling(252).max()
    signal_df['drawdown'] = (signal_df['Close'] - signal_df['high_52w']) / signal_df['high_52w']
    signal_df['drawdown_signal'] = signal_df['drawdown'] < drawdown_signal_threshold

    # === Vix Signal
    vix_close       = yf.Ticker('^VIX').history(period='max')['Close']
    vix_close.index = vix_close.index.tz_convert('US/Eastern')

    vix_close.index = vix_close.index.normalize()
    df.index        = df.index.normalize()

    signal_df = signal_df.merge(vix_close.rename('Close_vix'), left_index=True, right_index=True, how='left')
    signal_df['vix_signal'] = signal_df['Close_vix'] > vix_signal_threshold

    # === Breadth Thrust (% Stocks Above 50-Day MA)
    cache_is_stale = (
        not os.path.exists(CACHE) or
        datetime.fromtimestamp(os.path.getmtime(CACHE)) < datetime.now() - timedelta(days=STALE_AFTER_DAYS)
    )

    if cache_is_stale:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html = StringIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text)
        tickers = pd.read_html(html)[0]['Symbol'].tolist()
        ticker_data = yf.download(tickers=tickers, period='max')['Close']
        ticker_data.to_parquet(CACHE)
    else:
        ticker_data = pd.read_parquet(CACHE)

    ticker_data.index = ticker_data.index.tz_localize('US/Eastern')
    ticker_data.head()

    above_50d      = ticker_data > ticker_data.rolling(50).mean()
    breadth_thrust = above_50d.mean(axis=1).rename('breadth_thrust').to_frame()

    signal_df = signal_df.merge(breadth_thrust, how='left', on='Date')

    signal_df['breadth_thrust_signal'] = signal_df['breadth_thrust'] > breadth_thrust_signal_threshold


    # === Fear proxy: SPY 20-day return z-score (negative = fear)
    spy_ret = signal_df['Close'].pct_change(20)
    signal_df['fear_zscore'] = (spy_ret - spy_ret.rolling(252).mean()) / spy_ret.rolling(252).std()
    signal_df['fear_zscore_signal'] = signal_df['fear_zscore'] < fear_zscore_signal_threshold

    # === RSI Signal
    # Separate gains and losses
    delta = signal_df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = delta.where(delta < 0, 0).abs()

    # Calculate average gain and loss (Wilder smoothing)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()

    # Calculate RSI
    rs = avg_gain / avg_loss
    signal_df['RSI'] = 100 - (100 / (1 + rs))
    signal_df['RSI_signal'] = signal_df['RSI'] < rsi_signal_threshold

    # Composite signals 
    signal_df['composite_score'] = signal_df[[c for c in signal_df.columns if 'signal' in c]].sum(axis=1)

    return signal_df
