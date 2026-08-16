import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from bar_permute import get_permutation
from donchian import  walkforward_donch

df = pd.read_parquet('BTCUSD3600.pq')
df.index = df.index.astype('datetime64[s]')
df = df[(df.index.year >= 2016) & (df.index.year < 2021)]

df['r'] = np.log(df['close']).diff().shift(-1)

train_window = 24 * 365 * 4
df['donch_wf_signal'] = walkforward_donch(df, train_lookback=train_window)

donch_rets = df['donch_wf_signal'] * df['r']
real_wf_pf = donch_rets[donch_rets > 0].sum() / donch_rets[donch_rets < 0].abs().sum()

n_permutations = 200
perm_better_count = 1
permuted_pfs = []
print("Walkforward MCPT")
for perm_i in tqdm(range(1, n_permutations)):
    wf_perm = get_permutation(df, start_index=train_window)
    
    wf_perm['r'] = np.log(wf_perm['close']).diff().shift(-1) 
    wf_perm_sig = walkforward_donch(wf_perm, train_lookback=train_window)
    perm_rets = wf_perm['r'] * wf_perm_sig
    perm_pf = perm_rets[perm_rets > 0].sum() / perm_rets[perm_rets < 0].abs().sum()
    
    if perm_pf >= real_wf_pf:
        perm_better_count += 1

    permuted_pfs.append(perm_pf)


walkforward_mcpt_pval = perm_better_count / n_permutations
print(f"Walkforward MCPT P-Value: {walkforward_mcpt_pval}")


plt.style.use('dark_background')
pd.Series(permuted_pfs).hist(color='blue', label='Permutations')
plt.axvline(real_wf_pf, color='red', label='Real')
plt.xlabel("Profit Factor")
plt.title(f"Walkforward MCPT. P-Value: {walkforward_mcpt_pval}")
plt.grid(False)
plt.legend()
plt.show()

