import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet('BTCUSD3600.pq')
df.index = df.index.astype('datetime64[s]')

fast_ma = df['close'].rolling(10).mean()
slow_ma = df['close'].rolling(30).mean()

# Signal/position vector. The position at each bar
df['signal'] = np.where(fast_ma > slow_ma, 1, 0)

df['return'] = np.log(df['close']).diff().shift(-1)
df['strategy_return'] = df['signal'] * df['return']

r = df['strategy_return']
profit_factor = r[r>0].sum() / r[r<0].abs().sum()
sharpe_ratio = r.mean() / r.std()
