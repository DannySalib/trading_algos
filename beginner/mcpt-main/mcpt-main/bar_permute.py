import numpy as np
import pandas as pd
from typing import List, Union

def get_permutation(
    ohlc: Union[pd.DataFrame, List[pd.DataFrame]], start_index: int = 0, seed=None
):
    assert start_index >= 0

    np.random.seed(seed)

    if isinstance(ohlc, list):
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Indexes do not match"
        n_markets = len(ohlc)
    else:
        n_markets = 1
        time_index = ohlc.index
        ohlc = [ohlc]

    n_bars = len(ohlc[0])

    perm_index = start_index + 1
    perm_n = n_bars - perm_index

    start_bar = np.empty((n_markets, 4))
    relative_open = np.empty((n_markets, perm_n))
    relative_high = np.empty((n_markets, perm_n))
    relative_low = np.empty((n_markets, perm_n))
    relative_close = np.empty((n_markets, perm_n))

    for mkt_i, reg_bars in enumerate(ohlc):
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']])

        # Get start bar
        start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

        # Open relative to last close
        r_o = (log_bars['open'] - log_bars['close'].shift()).to_numpy()
        
        # Get prices relative to this bars open
        r_h = (log_bars['high'] - log_bars['open']).to_numpy()
        r_l = (log_bars['low'] - log_bars['open']).to_numpy()
        r_c = (log_bars['close'] - log_bars['open']).to_numpy()

        relative_open[mkt_i] = r_o[perm_index:]
        relative_high[mkt_i] = r_h[perm_index:]
        relative_low[mkt_i] = r_l[perm_index:]
        relative_close[mkt_i] = r_c[perm_index:]

    idx = np.arange(perm_n)

    # Shuffle intrabar relative values (high/low/close)
    perm1 = np.random.permutation(idx)
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]

    # Shuffle last close to open (gaps) seprately
    perm2 = np.random.permutation(idx)
    relative_open = relative_open[:, perm2]

    # Create permutation from relative prices
    perm_ohlc = []
    for mkt_i, reg_bars in enumerate(ohlc):
        perm_bars = np.zeros((n_bars, 4))

        # Copy over real data before start index 
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']]).to_numpy().copy()
        perm_bars[:start_index] = log_bars[:start_index]
        
        # Copy start bar
        perm_bars[start_index] = start_bar[mkt_i]

        for i in range(perm_index, n_bars):
            k = i - perm_index
            perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[mkt_i][k]
            perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
            perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
            perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]

        perm_bars = np.exp(perm_bars)
        perm_bars = pd.DataFrame(perm_bars, index=time_index, columns=['open', 'high', 'low', 'close'])

        perm_ohlc.append(perm_bars)

    if n_markets > 1:
        return perm_ohlc
    else:
        return perm_ohlc[0]

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    btc_real = pd.read_parquet('BTCUSD3600.pq')
    btc_real.index = btc_real.index.astype('datetime64[s]')
    btc_real = btc_real[(btc_real.index.year >= 2018) & (btc_real.index.year < 2020)]

    btc_perm = get_permutation(btc_real)

    btc_real_r = np.log(btc_real['close']).diff() 
    btc_perm_r = np.log(btc_perm['close']).diff()

    print(f"Mean. REAL: {btc_real_r.mean():14.6f} PERM: {btc_perm_r.mean():14.6f}")
    print(f"Stdd. REAL: {btc_real_r.std():14.6f} PERM: {btc_perm_r.std():14.6f}")
    print(f"Skew. REAL: {btc_real_r.skew():14.6f} PERM: {btc_perm_r.skew():14.6f}")
    print(f"Kurt. REAL: {btc_real_r.kurt():14.6f} PERM: {btc_perm_r.kurt():14.6f}")

    eth_real = pd.read_parquet('ETHUSD3600.pq')
    eth_real.index = eth_real.index.astype('datetime64[s]')
    eth_real = eth_real[(eth_real.index.year >= 2018) & (eth_real.index.year < 2020)]
    eth_real_r = np.log(eth_real['close']).diff()
    
    print("") 

    permed = get_permutation([btc_real, eth_real])
    btc_perm = permed[0]
    eth_perm = permed[1]
    
    btc_perm_r = np.log(btc_perm['close']).diff()
    eth_perm_r = np.log(eth_perm['close']).diff()
    print(f"BTC&ETH Correlation REAL: {btc_real_r.corr(eth_real_r):5.3f} PERM: {btc_perm_r.corr(eth_perm_r):5.3f}")

    plt.style.use("dark_background")    
    np.log(btc_real['close']).diff().cumsum().plot(color='orange', label='BTCUSD')
    np.log(eth_real['close']).diff().cumsum().plot(color='purple', label='ETHUSD')
    
    plt.ylabel("Cumulative Log Return")
    plt.title("Real BTCUSD and ETHUSD")
    plt.legend()
    plt.show()

    np.log(btc_perm['close']).diff().cumsum().plot(color='orange', label='BTCUSD')
    np.log(eth_perm['close']).diff().cumsum().plot(color='purple', label='ETHUSD')
    plt.title("Permuted BTCUSD and ETHUSD")
    plt.ylabel("Cumulative Log Return")
    plt.legend()
    plt.show()



